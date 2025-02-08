import pandas as pd
import re
import utils
from nltk.corpus import stopwords
from multiprocessing import Pool
import sys
sys.path.append("./py_scripts/")
import utils
import itertools
import numpy as np
import textstat

def crop_content_sections(text):
    start, end = utils.find_section_span(text, "content_sections")
    return text[start:end]

def get_stopword_frac(df, stop_words, section):
    
    with Pool(processes=48) as pool:
        df[f"stopword_count"] = pool.starmap(
            utils.count_stopword, 
            zip(df[section].astype(str).values, itertools.repeat(stop_words)))
    df["stopword_frac"] = df["stopword_count"] / df["note_len"]
    return df

def get_medicalword_frac(df, allow_list, section):
    with Pool(processes=48) as pool:
        df[f"medicalword_count"] = pool.starmap(
            utils.count_medicalword, 
            zip(df[section].astype(str).values, itertools.repeat(allow_list)))
    df["medicalword_frac"] = df["medicalword_count"] / df["note_len"]
    return df

def get_liwc_frac(df, liwc, liwc_categories, section):
    for cat in liwc_categories:
        with Pool(processes=48) as pool:
            df[f"{cat}_count"] = pool.starmap(
                utils.count_liwc_words, 
                zip(df[section].astype(str).values, itertools.repeat(liwc[cat])))
        df[f"{cat}_frac"] = df[f"{cat}_count"] / df["note_len"]
    return df

def get_readability_flesch_kincaid(df, section):
    with Pool(processes=48) as pool:
        df[f"readability_flesch_kincaid"] = pool.starmap(
            textstat.flesch_kincaid_grade, 
            zip(df[section].astype(str).values))
    return df

if __name__ == "__main__":

    allow_list = utils.get_allow_words()
    allow_list = set(allow_list)
    stop_words = stopwords.words('english')
    ppl_df = pd.read_csv("./data/lm_ppl/all_cohort_ppl_df.csv")

    liwc = utils.load_liwc()
    liwc_catetories = [
        "pronoun", "i", "we", "you", "shehe", "they", "ipron", #pronouns
        "affect", "posemo", "negemo", "anx", "anger", "sad", # affect
        "cogmech", "insight", "cause", "discrep", "tentat", "certain", "inhib", "incl", "excl", # cognitive
    ]

    train_df = pd.read_csv("./data/lm/train.csv")
    valid_df = pd.read_csv("./data/lm/validation.csv")
    test_df = pd.read_csv("./data/lm/test.csv")


    train_df = pd.concat([train_df, valid_df], axis=0)
    train_df = train_df.reset_index(drop=True)

    train_df["content_sections"] = train_df["comments"].apply(crop_content_sections)
    test_df["content_sections"] = test_df["comments"].apply(crop_content_sections)

    ppl_df[f"ppl_log"] = np.log(ppl_df[f"content_sections"])

    train_df = pd.merge(train_df, ppl_df[["ed_enc_id", "ppl_log"]], on="ed_enc_id")
    test_df = pd.merge(test_df, ppl_df[["ed_enc_id", "ppl_log"]], on="ed_enc_id")

    train_df["note_len"] = train_df["content_sections"].astype(str).apply(lambda x: len(x.split()))
    test_df["note_len"] = test_df["content_sections"].astype(str).apply(lambda x: len(x.split()))

    print("Stop words")
    train_df = get_stopword_frac(train_df, stop_words, "content_sections")
    test_df = get_stopword_frac(test_df, stop_words, "content_sections")

    print("Medical words")
    train_df = get_medicalword_frac(train_df, allow_list, "content_sections")
    test_df = get_medicalword_frac(test_df, allow_list, "content_sections")

    print("LIWC")
    train_df = get_liwc_frac(train_df, liwc, liwc_catetories, "content_sections")
    test_df = get_liwc_frac(test_df, liwc, liwc_catetories, "content_sections")


    print("Readability")
    train_df = get_readability_flesch_kincaid(train_df, "content_sections")
    test_df = get_readability_flesch_kincaid(test_df, "content_sections")


    print(train_df.shape)
    print(test_df.shape)

    # describe float type columns
    print(train_df.describe().T)
    print(test_df.describe().T)

    train_df.to_csv("./data/train.csv", index=False)
    test_df.to_csv("./data/test.csv", index=False)


