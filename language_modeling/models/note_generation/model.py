
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
import argparse
import os
from pathlib import Path
from typing import Any, Dict
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, f1_score, average_precision_score, classification_report
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import json
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

from models.note_generation.dataloader import TransformerDataset
from models.note_generation.utils import top_k_top_p_filtering

from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}


# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


class Transformer_PL(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        num_labels=None,
        config=None,
        tokenizer=None,
        model=None,
        **config_kwargs
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()
        # TODO: move to self.save_hyperparameters()
        # self.save_hyperparameters()
        # can also expand arguments into trainer signature for easier reading
        # mode="sequence-classification"
        mode = "language-modeling"

        self.save_hyperparameters(hparams)
        logger.info(f"Number of Labels: {self.hparams.num_labels}")
        self.step_count = 0
        self.output_dir = Path(self.hparams.output_dir)
        cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.hparams.config_name if self.hparams.config_name else self.hparams.model_name_or_path,
                # **({"num_labels": self.hparams.num_labels} if self.hparams.num_labels is not None else {}),
                cache_dir=cache_dir,
                # **config_kwargs,
            )
            print(self.config)
        else:
            self.config: PretrainedConfig = config

        extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
        for p in extra_model_params:
            if getattr(self.hparams, p, None):
                assert hasattr(self.config, p), f"model config doesn't have a `{p}` attribute"
                setattr(self.config, p, getattr(self.hparams, p))

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_name if self.hparams.tokenizer_name else self.hparams.model_name_or_path,
                cache_dir=cache_dir,
            )
            special_tokens_dict = {'pad_token': '<PAD>'}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
            # self.tokenizer.pad_token = self.tokenizer.eos_token

        else:
            self.tokenizer: PreTrainedTokenizer = tokenizer
        self.model_type = MODEL_MODES[mode]
        if model is None:
            self.model = self.model_type.from_pretrained(
                self.hparams.model_name_or_path,
                from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
                config=self.config,
                cache_dir=cache_dir, # save server storage
            )
        else:
            self.model = model
        self.model.resize_token_embeddings(len(self.tokenizer))

        if self.hparams.structured:
            structured_length = 27216 # tentitively
            self.hparams.structured_length = structured_length
            self.feature_encoder = FeatureEncoder(self.config, self.hparams)
        if self.hparams.doctor:
            self.doctor_encoder = DoctorEncoder(self.config, self.hparams)

    def load_hf_checkpoint(self, *args, **kwargs):
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams.adafactor:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
            )
        self.opt = optimizer

        self.train_loader = getattr(self,"train_loader",None)
        if self.train_loader:
            scheduler = self.get_lr_scheduler()
        else:
            return [optimizer]
        return [optimizer], [scheduler]

    def forward(self, **inputs):
        labels = inputs.pop('labels')
        input_ids = inputs.pop('input_ids')
        attention_mask = inputs.pop('attention_mask')
        structured_features = inputs.pop('structured')
        doctor_ids = inputs.pop('doctor_ids')

        note_embeds = self.model.transformer.wte(input_ids[...,:-1].contiguous()) #return [B, L-1, D]
        attention_mask = attention_mask[...,:-1].contiguous()

        if self.hparams.structured:
            features = self.feature_encoder(structured_features).unsqueeze(1) #return [B, 1, D]
            note_embeds = note_embeds + features
        if self.hparams.doctor:
            doctor_vectors = self.doctor_encoder(doctor_ids) #return [B, 1, D]
            note_embeds = note_embeds + doctor_vectors 
        
        outputs = self.model(inputs_embeds=note_embeds, attention_mask=attention_mask) #[B, L-1, D]
        logits = outputs.logits
        target_input_ids = input_ids[:,1:].contiguous() #return [B, L-1]
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(logits.view(-1, logits.size(-1)), target_input_ids.view(-1))
        return loss, logits

    def generate(self, structured_feature, doctor_id, hint_ids=None, max_generate_length=512):
                
        """ Predict function.
        :param sample: dictionary with the text we want to classify.
        :param max_length: maximum length of the sequence to be generated.
        :param hint_ids: list of tokens that are first 10 tokens of the groud truth note.
        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            features = self.feature_encoder(structured_feature) #return [1, D]
            doctor_vectors = self.doctor_encoder(doctor_id).squeeze(0) #return [1, D]
            input_embeds = torch.cat([features, doctor_vectors], dim=0).contiguous() #return [2, D]

            output_seq = torch.Tensor([0]).to(self.device)
            predicted_token = torch.Tensor([0]).to(self.device)

            if hint_ids is not None:
                input_embeds = torch.cat([input_embeds, self.model.transformer.wte(hint_ids)], dim=0).contiguous()
                output_seq = torch.cat([output_seq, hint_ids], dim=0).contiguous()
                predicted_token = torch.cat([predicted_token, hint_ids], dim=0).contiguous()

            while (
                predicted_token.unsqueeze(-1)[0] != self.tokenizer.pad_token_id
                and len(output_seq) < max_generate_length
            ):
                outputs = self.model(inputs_embeds=input_embeds)
                lm_logits = outputs.logits
                logits = lm_logits[-1, :]
                top_k, top_p, temperature = 0, 0.95, 1
                filtered_logits = top_k_top_p_filtering(
                    logits, top_k=top_k, top_p=top_p, temperature=temperature
                )
                probabilities = torch.softmax(filtered_logits, dim=-1)
                probabilities_logits, probabilities_position = torch.sort(
                    probabilities, descending=True
                )
                predicted_token = torch.multinomial(probabilities, 1)
                output_seq = torch.cat([output_seq, predicted_token])
                next_token_embeds = self.model.transformer.wte(predicted_token)
                input_embeds = torch.cat([input_embeds, next_token_embeds], dim=0).contiguous()
                # logger.info(output_seq)
            output_seq = (
                output_seq[1:-1]
                if predicted_token.unsqueeze(-1)[0] == self.tokenizer.pad_token_id
                else output_seq[1:]
            )
            output_seq = output_seq.cpu().numpy().astype(int)
            output_sentence = self.tokenizer.decode(output_seq)
            # logger.info(output_seq)
            # logger.info(f"generated sentence: {output_sentence}")

        if self.training:
            self.train()
        return output_sentence



    def training_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "structured": batch[5], "doctor_ids": batch[6]}

        # if batch_idx % 50 == 0:
        #     self.generate(**inputs)

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None

        outputs = self(**inputs)
        loss= outputs[0]

        lr_scheduler = self.trainer.lr_schedulers[0]["scheduler"]
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log( "rate", lr_scheduler.get_last_lr()[-1])
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "structured": batch[5], "doctor_ids": batch[6]}
        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        # logger.info(tmp_eval_loss)
        # self.log('val_loss', tmp_eval_loss.item())
        return {"loss":tmp_eval_loss.item(), "labels":batch[3], "stays":batch[4]}

    def validation_epoch_end(self, outputs):
        mean_loss = np.mean([x["loss"] for x in outputs if not np.isnan(x["loss"])])
        print(mean_loss)
        self.log("val_loss", mean_loss)
        return #self.validation_end(outputs)

    def test_step(self, batch, batch_nb):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "structured": batch[5], "doctor_ids": batch[6]}

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.config.model_type in ["bert", "xlnet", "albert"] else None

        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        self.log('test_loss', tmp_eval_loss.item())
        return {"loss":tmp_eval_loss.item(), "labels":batch[3], "stays":batch[4]}

    def test_epoch_end(self, outputs):
        mean_loss = np.mean([x["loss"] for x in outputs if not np.isnan(x["loss"])])
        self.log("test_loss", mean_loss)
        return #self.validation_end(outputs)
    
    def vote_score(self, outputs, mode):
        all_probs, all_labels, all_stays = [],[],[]
        for output in outputs:
            probs = list(torch.softmax(output['logits'], dim=1)[:,1].cpu().detach().numpy()) # positive probs
            labels = list(output['labels'].flatten().cpu().detach().numpy())
            stays = list(output['stays'])
            all_probs.extend(probs)
            all_labels.extend(labels)
            all_stays.extend([stay.split('.')[0] for stay in stays])
        df = pd.DataFrame({
            "ID": all_stays,
            "pred_score": all_probs,
            "Label": all_labels
        })
        """
        credit to https://github.com/kexinhuang12345/clinicalBERT/blob/master/run_readmission.py 
        """
        df_sort = df.sort_values(by=['ID'])
        df_sort.to_csv(self.trainer.checkpoint_callback.dirpath+f"/{mode}_epoch{self.current_epoch}_result.csv")
        #score 
        temp = (df_sort.groupby(['ID'])['pred_score'].agg(max)+df_sort.groupby(['ID'])['pred_score'].agg(sum)/2)/(1+df_sort.groupby(['ID'])['pred_score'].agg(len)/2)
        x = df_sort.groupby(['ID'])['Label'].agg(np.min).values
        df_out = pd.DataFrame({'ID': df_sort['ID'].unique(), 'logits': temp.values, 'label': x})
        df_out.to_csv(self.trainer.checkpoint_callback.dirpath+f"/{mode}_epoch{self.current_epoch}_reduced_result.csv")


        roc_auc = roc_auc_score(x, temp.values)
        pr_auc = average_precision_score(x, temp.values)
        f1 = f1_score(x, temp.values>=0.5)
        fpr, tpr, thresholds = roc_curve(x, temp.values)
        auc_score = auc(fpr, tpr)

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='{}-(area = {:.3f})'.format(mode, auc_score))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()
        string = 'auroc_clinicalbert_'+f"{mode}_epoch{self.current_epoch}"+'.png'
        plt.savefig(os.path.join(self.trainer.checkpoint_callback.dirpath , string))

        return roc_auc, pr_auc, f1

    @property
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        dataset_size = len(self.train_loader.dataset)
        return (dataset_size / effective_batch_size) * self.hparams.max_epochs

    def setup(self, stage):
        if stage == "fit":
            self.train_loader = self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)

    def get_dataloader(self, type_path, batch_size, shuffle=False):
        # todo add dataset path
        # filename = f'{self.hparams.note_type}_seg_note_{type_path}_{self.hparams.period}.csv'
        type_path = "validation" if type_path == "valid" else type_path
        filename = f"{self.hparams.data_dir}/{type_path}.csv"

        if self.hparams.structured:
            structured_name = f"{type_path}_generation.pkl"
            if self.hparams.balanced:
                structured_filepath = os.path.join(self.hparams.data_dir, 'structured_features', 'balanced', structured_name)
            else:
                structured_filepath = os.path.join(self.hparams.data_dir, 'structured_features', structured_name)
            logger.info(f"Loading structured features at: /data/joe/note_generation/features_retro_note_generation.pkl")
            structured_features = pickle.load(open(structured_filepath, "rb"))
        else:
            structured_features = None

        data = pd.read_csv(filename)
        yelp = TransformerDataset(
            self.tokenizer, data, self.hparams.max_seq_length, structured_features)
        logger.info(f"Loading {type_path} dataset with length {len(yelp)} from {filename}")
        data_loader = torch.utils.data.DataLoader(dataset=yelp,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=self.hparams.num_workers,
                                                collate_fn=yelp.collate_fn)
        
        return data_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.get_dataloader("valid", self.hparams.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)


    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("best_tfmr")
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--num_labels",
            default=2,
            type=int,
            help="number of labels to generate",
        )
        parser.add_argument(
            "--num_doctors",
            default=62,
            type=int,
            help="number of labels to generate",
        )
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument(
            "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="Encoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--decoder_layerdrop",
            type=float,
            help="Decoder layer dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            help="Dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument(
            "--attention_dropout",
            type=float,
            help="Attention dropout probability (Optional). Goes into model.config",
        )
        parser.add_argument("--learning_rate", default=5e-6, type=float, help="The initial learning rate for Adam.")
        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            choices=arg_to_scheduler_choices,
            metavar=arg_to_scheduler_metavar,
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--num_workers", default=16, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--num_train_epochs", dest="max_epochs", default=3, type=int)
        parser.add_argument("--train_batch_size", default=4, type=int)
        parser.add_argument("--eval_batch_size", default=4, type=int)
        parser.add_argument("--adafactor", action="store_true")
        parser.add_argument("--structured", action="store_true")
        parser.add_argument("--doctor", action="store_true")

        return parser


class FeatureEncoder(nn.Module):
    """Encode structured vaiables into language model space."""

    def __init__(self, config, args):
        super().__init__()

        self.linear1 = nn.Linear(args.structured_length, config.n_embd)
        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.linear2 = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(0.15)
        self.gelu = nn.GELU()

    def forward(self, structured, **kwargs):
        hidden_states = self.linear1(structured)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.linear2(hidden_states)
        return output

class DoctorEncoder(nn.Module):
    """Encode structured vaiables into language model space."""

    def __init__(self, config, args):
        super().__init__()

        self.embedding = nn.Embedding(args.num_doctors, config.n_embd)

    def forward(self, doctor_ids, **kwargs):
        hidden_states = self.embedding(doctor_ids)
        return hidden_states
