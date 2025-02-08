import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import pandas as pd
from transformers import AutoModelWithLMHead, AutoTokenizer
import re
from bisect import bisect_left, bisect_right
import sys
import numpy as np
import pickle

dfs = []
dfs.append(pd.read_csv("./data/lm/train.csv"))
dfs.append(pd.read_csv("./data/lm/validation.csv"))
dfs.append(pd.read_csv("./data/lm/test.csv"))

df = pd.concat(dfs, axis=0)

tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="./data/transformers")

def find_section_span(text, section):
    start = 0
    total_len = len(text)
    content_end = text.find("________")
    if content_end != -1:
        text = text[:content_end]
    end = len(text)
    
    flag = False
    if section == "content_sections":
        for m in re.finditer(r"\n\n([A-Z][A-Z][A-Z\s]+):", text):
            if not flag:
                start = m.start(0)
                flag = True
                break
        if flag:
            return start, end
        else:
            return None, None
    elif section == "drop_end":
        return start, end
    elif section == "keep_end":
        if content_end != -1:
            return content_end, total_len
        else:
            return start, end
    else:
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

def compute_target_section_perplexity(note, ed_enc_id, target_sections):
    
    output_dicts = []
    
    #print(f"\rCurrent md id : {d_md_id}", end="")
    output_dict = {"ed_enc_id":ed_enc_id}

    # list with len [n_token-1] since first token doesn't have loss
    note_token_loss = pickle.load(
        open(f"./data/lm_ppl/token_loss/comments_{ed_enc_id}.pkl","rb")
    )
    inputs = tokenizer.encode_plus(note, return_offsets_mapping=True,)
    assert len(note_token_loss) == ( len(inputs['offset_mapping'])-1)

    for target_section in target_sections:
        if target_section == "comments":
            token_loss = note_token_loss
        else:
            start_idx, end_idx = find_section_span(note, target_section)
            if start_idx is None:
                output_dict[target_section] = None
                continue
            start_token_idx = bisect_right([r for l, r in inputs['offset_mapping']], start_idx)
            end_token_idx = bisect_left([l for l, r in inputs['offset_mapping']], end_idx)

            """
            note token loss is 1d array [n_token-1] since first token doesn't have loss, so we need to minus 1 to get the right span of text
            for example, A B C D E F G, we want to get the loss of B C D, so we need to get the loss of token idx 0, 1, 2
            since the the loss array represents the loss of B C D E F G only.
            However, to get the original text, we need to get the text of token idx 1, 2, 3.
            """
            token_loss = note_token_loss[max(start_token_idx-1, 0):end_token_idx-1]
            
        token_loss = np.array(token_loss)
        token_loss = token_loss[~np.isnan(token_loss)]
        output_dict[target_section] = np.exp(np.mean(token_loss))
    output_dicts.append(output_dict)
    return output_dicts

target_sections = [
        # Removing beginning and ending sections, e.g., edvist and trailing signature
        "content_sections",
        ]

from multiprocessing import Pool
from itertools import repeat


# use pool to speed up
with Pool(48) as pool:
    output_ppls = pool.starmap(
        compute_target_section_perplexity,
        zip(df.comments.values, df.ed_enc_id.values, repeat(target_sections))
    )

# unwrap the list of list
output_ppls = [item for sublist in output_ppls for item in sublist]

print(output_ppls[:10])
# commented out for now to generate token loss for the whole notes
ppl_df = pd.DataFrame(output_ppls)
ppl_df.to_csv("./data/lm_ppl/all_cohort_ppl_df.csv", index=False)
