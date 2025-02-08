"""
Run scripts in language_modeling first to get the fine-tuned model.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import pandas as pd
import torch
import sys
import numpy as np
sys.path.append('./language_modeling')
from models.note_generation.model import Transformer_PL
import pytorch_lightning as pl
import pickle
import time
pl.seed_everything(42)
device = "cpu"

pl_model = Transformer_PL.load_from_checkpoint(
    "./data/lm/exp/lm-comments-5e-05-gpt2/checkpoints/epoch=4-val_loss=1.06.ckpt", 
    cache_dir="./data/transformers"
)
pl_model.to(device)
pl_model.eval()
pl_model.freeze()
if device == "cuda":
    pl_model.half()

cohort_note = pd.read_csv("./data/lm/test.csv")

def convert_seqs_to_overlapped_token(text, max_len=1024, stride=512, ignore_index=-100):
    texts = pl_model.tokenizer([text])
    concat_tokens = [t for tokens in texts.input_ids for t in tokens]
    # print("total tokens", len(concat_tokens))
    # pad concat_tokens length to multiple of self.max_len
    attention_masks = [1] * len(concat_tokens) + [0] * (max_len - len(concat_tokens) % max_len)
    concat_tokens = concat_tokens + [0] * (max_len - len(concat_tokens) % max_len)
    
    # reshape to [-1, self.max_len]
    input_ids, attention_mask = [], []
    for i in range(0, len(concat_tokens), stride):
        input_ids.append(concat_tokens[i:i+max_len])
        attention_mask.append(attention_masks[i:i+max_len])
        if i+max_len >= len(concat_tokens):
            break
    return torch.LongTensor(input_ids), torch.LongTensor(attention_mask)


def get_note_token_loss(pl_model, note, md_id=None):
    input_ids, attention_mask = convert_seqs_to_overlapped_token(note)
    # to device
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    inputs = {"input_ids": input_ids,
              "attention_mask": attention_mask,
              "labels": None,
              "structured": None,
              "doctor_ids": torch.LongTensor([[md_id]]).to(device) if md_id is not None else None,}

    with torch.no_grad():
        loss, logits = pl_model(**inputs)
        loss = torch.nn.CrossEntropyLoss(reduction="none")(
            logits.view(-1, logits.size(-1)), inputs["input_ids"][:,1:].contiguous().view(-1)
            )
    loss = loss.view(-1, 1023)
    # return 1D token loss
    token_losses = []
    if len(loss) <= 1:
        tmp_loss, tmp_att_mask = loss[0], attention_mask[0][1:]
        tmp_loss = tmp_loss[tmp_att_mask==1]
        return tmp_loss.to("cpu").numpy().tolist()
    else:
        token_losses.append(loss[0])
        for i in range(1, len(loss)):
            tmp_loss, tmp_att_mask = loss[i][-512:], attention_mask[i][-512:]
            assert len(tmp_loss) == len(tmp_att_mask)
            tmp_loss = tmp_loss[tmp_att_mask==1]
            token_losses.append(tmp_loss)
        return torch.cat(token_losses, dim=0).to("cpu").numpy().tolist()

for i, row in cohort_note.iterrows():
    if i % 100 == 0:
        print(i, time.ctime())
    # doc = nlp(row.comments) # for sentence processing
    inputs = pl_model.tokenizer.encode_plus(
        row.comments, return_offsets_mapping=True
    )
    loss = get_note_token_loss(pl_model, row.comments, md_id=None)
    assert len(loss) == ( len(inputs['offset_mapping'])-1) # loss is one less than offset_mapping becuase there's no prob for the first token
    pickle.dump(
        loss,
        open(f"./data/lm_ppl/token_loss/comments_{row.ed_enc_id}.pkl","wb")
    )
