import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import pandas as pd
from transformers import AutoModelWithLMHead, AutoTokenizer
from nltk.tokenize import sent_tokenize
import torch
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import re
from bisect import bisect_left, bisect_right
import pickle
import seaborn as sns

import sys
import numpy as np
sys.path.append("./language_modeling')
from models.note_generation.model import Transformer_PL
import pytorch_lightning as pl

def find_section_span(text, section):
    start, end = 0, len(text)
    flag = False
    for m in re.finditer(r"\n\n([A-Z][A-Z][A-Z\s]+):", text):
        if text[m.start(0): m.end(0)] == f"\n\n{section}:":
            start = m.end(0)
            flag = True
        elif flag:
            end = m.start(0)
            break
    if flag:
        return start, end
    else:
        return None, None

def get_prompt_text(note, target_section):
    inputs = pl_model.tokenizer.encode_plus(note, return_offsets_mapping=True,)
    start_idx, end_idx = find_section_span(note, target_section)
    if start_idx is None:
        return None
    start_token_idx = bisect_right([r for l, r in inputs['offset_mapping']], start_idx)
    end_token_idx = bisect_left([l for l, r in inputs['offset_mapping']], end_idx)

    section_text = pl_model.tokenizer.decode(
        inputs["input_ids"][start_token_idx:end_token_idx]
    )
    if start_token_idx > 768:
        init_token_idx = start_token_idx - 768
    else:
        init_token_idx = 0
        
    p_text = pl_model.tokenizer.decode(
        inputs["input_ids"][init_token_idx:start_token_idx]
    )
    return p_text, section_text


def prompt_model(prompt, do_sample=False, topp=0.97, topk=None):
    inputs = pl_model.tokenizer.encode_plus(prompt, return_tensors="pt").to("cpu")
    if do_sample:
        return pl_model.tokenizer.decode(pl_model.model.generate(**inputs, max_new_tokens=256, do_sample=True, topp=topp, topk=topk)[0])
    else:
        return pl_model.tokenizer.decode(pl_model.model.generate(**inputs, max_new_tokens=256)[0])

pl.seed_everything(42)

device = "cpu"
pl_model = Transformer_PL.load_from_checkpoint("./data/lm/exp/lm-comments-5e-05-gpt2/checkpoints/epoch=4-val_loss=1.06.ckpt", 
                                                    cache_dir="./data/transformers")
pl_model.to(device)
pl_model.eval()
pl_model.freeze()
if device != "cpu":
    pl_model.half()

df = pd.read_csv("./data/lm/test.csv")

output_dir = "./data/"
cache_dir = "./data/cache_note_completion_notopk"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    


for section in ["HISTORY OF PRESENT ILLNESS", ]:
    sec_df = df.dropna(subset = [section])
    for i, row in sec_df.reset_index(drop=True).iterrows():
        
        # pass if already generated in cache
        if os.path.exists(f"{cache_dir}/{section}/{row.ed_enc_id}.csv"):
            continue
        
        tmp_dict = {
            "section": section,
            "ed_enc_id": row.ed_enc_id,
            "note": row.comments,
        }
        
        if i%10 ==0:
            print(f"\r processing {i} notes with {section}", end="")
        note = row.comments
        prompt_text, section_text = get_prompt_text(note, section)
        
        for topp in [1.0]:   #0.8, 0.9, 
            sampled_output = prompt_model(prompt_text, do_sample=True, topp=topp)
            tmp_dict.update({
            f"sampled_note_topp={topp}": sampled_output,
        })
        tmp_df = pd.DataFrame([tmp_dict])
        # create cache dir if not exist
        
        # create section dir if not exist
        if not os.path.exists(os.path.join(cache_dir, section)):
            os.makedirs(os.path.join(cache_dir, section))
        
        # save to cache section dir
        tmp_df.to_csv(os.path.join(cache_dir, section, f"{row.ed_enc_id}.csv"), index=False)
    
    # merge all cache files to one csv
    generated_notes = []
    for file in os.listdir(os.path.join(cache_dir, section)):
        cache_df = pd.read_csv(os.path.join(cache_dir, section, file))
        cache_df = cache_df[["ed_enc_id", "section", "note", "sampled_note_topp=1.0"]]
        generated_notes.append(cache_df)
    generated_notes = pd.concat(generated_notes)
    generated_notes.to_csv(os.path.join(output_dir, f"note_completion_notopk_1.0only_{section}.csv"), index=False)
    
        