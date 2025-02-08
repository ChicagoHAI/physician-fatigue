import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
from datetime import datetime
import argparse
import glob
import os
import shutil
import time
from argparse import Namespace
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Dict, List

from transformers.data.processors.utils import InputFeatures
from sklearn.metrics import f1_score

from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.plugins.environments import SLURMEnvironment
from models.note_generation.model import Transformer_PL
import wandb
logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                rank_zero_info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Test results *****")
        metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))

def add_generic_args(parser, root_dir) -> None:
    #  TODO(SS): allow all pl args? parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--offline", action="store_true", default=False, help="Whether to upload to wandb.")

    parser.add_argument(
        "--max_epochs",
        default=10,
        type=int,
        help="The number of GPUs allocated for this, it is by default 0 meaning none",
    )
    parser.add_argument(
        "--min_epochs",
        default=5,
        type=int,
        help="The number of GPUs allocated for this, it is by default 0 meaning none",
    )

    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="The number of GPUs allocated for this, it is by default 0 meaning none",
    )
    parser.add_argument(
            "--max_seq_length",
            default=2000,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--fp16",default=False,
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int)
    parser.add_argument("--max_grad_norm", dest="gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--do_train", action="store_true", default=False, help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files.",
    )
    parser.add_argument(
        "--dataset",
        default="bwh",
        type=str,
        help="Pretrained tokenizer name or path",
    )
    parser.add_argument(
        "--task",
        choices=['lm'],
        type=str,
        help="Pretrained tokenizer name or path",
    )
    parser.add_argument(
        "--note_type",
        type=str, 
        choices=['comments', ],
        help="Directory where the created data should be stored."
    )
    parser.add_argument("--balanced", action="store_true", default=False, help="train on balanced dataset.")
    parser.add_argument("--overwrite_dir", action="store_true", default=False, help="overwrite existing experiment dir.")
    
    
def generic_train(
    model: Transformer_PL,
    args: argparse.Namespace,
    early_stopping_callback=False,
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs
):
    
    # init model
    if args.do_train:
        odir = Path(model.hparams.output_dir)
        odir.mkdir(exist_ok=True)
    log_dir = Path(os.path.join(model.hparams.output_dir, 'logs'))
    log_dir.mkdir(exist_ok=True)

    # build logger
    ## WanDB logger
    model_name = args.model_name_or_path.split("/")[-1]
    task_id = f"{args.task}-{args.note_type}-{args.learning_rate}-{model_name}"
    
    if args.structured:
        task_id += "-structured"
    if args.doctor:
        task_id += "-doctor"
    if args.balanced:
        task_id += "-balanced"


    if args.do_train:
        exp_dir = os.path.join(args.output_dir, task_id)
        if os.path.exists(exp_dir) and not args.overwrite_dir:
            logger.error("Output directory already exists! Use `overwrite_dir` to overwrite existing dir.")
            raise("Output directory already exists! Use `overwrite_dir` to overwrite existing dir.")
        else:
            if os.path.exists(exp_dir):
                shutil.rmtree(exp_dir)
            odir = Path(exp_dir)
            odir.mkdir()

    # sweep_id = os.environ.get(wandb.env.SWEEP_ID)
    # run_id = os.environ.get(wandb.env.RUN_ID)
    # if args.local_rank in ["-1", "0", 0, -1]:
    #     experiment = wandb.init(project="investigate-clinicalbert", entity="joe32140")
    #     pl_logger = pl_loggers.WandbLogger(
    #         project=f"info-solicitation",
    #         id=task_id,
    #         version=task_id,
    #         experiment=experiment,
    #         offline=args.offline
    #     )
    #     logger_version = task_id
    # else:
    #     pl_logger = None
    #     logger_version = "not_rank_0"

    # Tensorboard logger
    if args.local_rank in ["-1", "0", 0, -1]:
        pl_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_dir,
        version=task_id,
        name="",
        default_hp_metric=True
        )
        logger_version = task_id
    else:
        pl_logger = None
        logger_version = "not_rank_0"

    # add custom checkpoints
    ckpt_path = os.path.join(
        args.output_dir, logger_version, "checkpoints",
    )

    ckpt_dir = Path(ckpt_path)
    ckpt_dir.parent.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    logger.info(f"Checkpoint path: {ckpt_path}")
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_path, filename="{epoch}-{val_loss:.2f}", monitor="val_loss", mode="min", save_top_k=1, verbose=True
        )
    if logging_callback is None:
        logging_callback = LoggingCallback()

    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=True, mode="min")

    train_params = {}

    # TODO: remove with PyTorch 1.6 since pl uses native amp
    train_params["max_epochs"] = args.max_epochs
    train_params["min_epochs"] = args.min_epochs
    
    if args.fp16:
        train_params["precision"] = 16
        # train_params["amp_level"] = args.fp16_opt_level

    if args.gpus > 1:
        train_params["strategy"] = "ddp"

    train_params["accumulate_grad_batches"] = args.accumulate_grad_batches
    

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[logging_callback, checkpoint_callback, early_stop_callback] + extra_callbacks,
        logger=pl_logger,
        weights_save_path=ckpt_path,
        # plugins=[SLURMEnvironment(auto_requeue=True)],
        **train_params,
    )
    logger.info(os.environ)
    ## Need to upgrade pytorch lightning to enable lr find
    # if args.lr_find:
    #     lr_finder = trainer.lr_find(model)
    #     fig = lr_finder.plot(suggest=True)
    #     fig.show()
    #     new_lr = lr_finder.suggestion()
    #     logger.info("Recommended Learning Rate: %s", new_lr)

    if args.do_train:
        trainer.fit(model)
        # track model performance under differnt hparams settings in "Hparams" of TensorBoard
        # pl_logger.log_hyperparams(params=model.hparams, metrics={'hp_metric': checkpoint_callback.best_model_score.item()})
        # pl_logger.save()
    return trainer



def main():
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = Transformer_PL.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    print(args)
    # fix random seed to make sure the result is reproducible
    pl.seed_everything(args.seed)

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "./results",
            f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(args.output_dir)

    model = Transformer_PL(args)
    trainer = generic_train(model, args)
    # Optionally, predict on dev set and write to output_dir
    if args.do_predict:
        logger.info("Testing!")
        checkpoints = list(sorted(glob.glob(os.path.join(trainer.checkpoint_callback.dirpath, "epoch=*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1], structured=args.structured)
        
        return trainer.test(model)

if __name__ == "__main__":
    main()
