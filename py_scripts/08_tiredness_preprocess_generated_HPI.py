import pandas as pd
import re
from nltk.corpus import stopwords
from multiprocessing import Pool
import sys
sys.path.append("./py_scripts/")
import utils
import itertools
import numpy as np
import textstat

def find_content_sections(text):
    start = 0
    content_end = text.find("________")
    if content_end != -1:
        text = text[:content_end]
    end = len(text)
    
    flag = False
    for m in re.finditer(r"\n\n([A-Z][A-Z][A-Z\s]+):", text):
        if not flag:
            start = m.start(0)
            flag = True
            break
    if flag:
        return text[start:end]
    else:
        return ""
    
def crop_content_sections(text):
    start, end = utils.find_section_span(text, section)
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

    liwc = utils.load_liwc()
    liwc_catetories = [
        "pronoun", "i", "we", "you", "shehe", "they", "ipron", #pronouns
        "affect", "posemo", "negemo", "anx", "anger", "sad", # affect
        "cogmech", "insight", "cause", "discrep", "tentat", "certain", "inhib", "incl", "excl", # cognitive
    ]

    df = pd.read_csv("./data/tiredness_generated_HPI.csv")
    section = "HISTORY OF PRESENT ILLNESS"
    
    df[f"ppl_log"] = df["log_ppl"]

    df["note_len"] = df[section].astype(str).apply(lambda x: len(x.split()))

    print("Stop words")
    df = get_stopword_frac(df, stop_words, section)

    print("Medical words")
    df = get_medicalword_frac(df, allow_list, section)

    print("LIWC")
    df = get_liwc_frac(df, liwc, liwc_catetories, section)


    print("Readability")
    df = get_readability_flesch_kincaid(df, section)

    print(df.shape)

    # describe float type columns
    print(df.describe().T)

    df.to_csv("./data/generated_HPI_processed.csv", index=False)
