import pandas as pd
import numpy as np
import pickle
import json
import re
import collections

from nltk import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, average_precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import normalize

from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from scipy.sparse import csr_matrix
import numpy as np
import os
from collections import Counter
import bisect
from matplotlib import pyplot as plt
import functools

def precision_at_k_percent(y_label, y_pred, k=10):
    pairs = list(zip(y_label, y_pred))
    pairs.sort(key=lambda x:x[1], reverse=True)
    top_pairs = pairs[:int((len(pairs)/100)*k)]
    return sum(p[0] for p in top_pairs) / len(top_pairs)

def precision_at_k(y_label, y_pred, k=100):
    pairs = list(zip(y_label, y_pred))
    pairs.sort(key=lambda x:x[1], reverse=True)
    top_pairs = pairs[:k]
    return sum(p[0] for p in top_pairs) / k

def get_allow_words():
    # obtain allowed words from snomed ontology
    with open("./data/concept_description_less_than_three_tokens.txt") as f:
        allow_words = f.read().splitlines()
    return allow_words


def find_section_span(text, section):
    start = 0
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

def get_note_sections(training_cohort_note, note_column):
    section_counter = Counter()
    for i, row in training_cohort_note.iterrows():
        section_counter.update(
            re.findall("\n\n([A-Z][A-Z][A-Z\s]+):", row[note_column])
        )
    selected_section_titles = ["EDVISIT"]
    print(section_counter.most_common(49))
    selected_section_titles.extend(s[0] for s in section_counter.most_common(49))
    section_set = set(selected_section_titles)
    return section_set

def segment_note2sections(cohort_note, section_set, note_column):
    sections2text = {s:[] for s in section_set}

    for i, row in cohort_note.iterrows():
        test = re.sub(r"\n\n([A-Z][A-Z][A-Z\s]+):", r'\n\n\n\1:', row[note_column])
        test = test[:test.find("________________")]
        test = re.sub("\n\n", ' ', test)
        test = re.sub("EDVISIT\^\d+\^\w+\, \w+\^\d+/\d+/\d+\^\w+\, \w+","EDVISIT:", test)
        
        sec2text = {}
        for section in test.split("\n"):
            match = re.match("([\w\W]+): ([\w\W]+)", section)
            if match:
                g = match.groups()
                if g[0] in section_set:
                    sec2text[g[0]] = g[1].strip()
                    
        for key, value in sections2text.items():
            value.append(sec2text.get(key, None))
    return pd.DataFrame(sections2text)
        
def count_stopword(text, stop_words):
    return len([word for word in text.lower().split() if word in stop_words])

def count_medicalword(text, medical_words):
    return len([word for word in text.lower().split() if word in medical_words])


def load_liwc(filename="./data/LIWC2007_English100131.dic"):
    """
    Load LIWC dataset
    input: a file that stores LIWC 2007 English dataset
    output:
        result: a dictionary that maps each word to the LIWC cluster ids
            that it belongs to
        class_id: a dict that maps LIWC cluster to category id,
            this does not seem useful, here for legacy reasons
        cluster_result: a dict that maps LIWC cluster id to all words
            in that cluster
        categories: a dict that maps LIWC cluster to its name
        category_reverse: a dict that maps LIWC cluster name to its id
    """
    # be careful, there can be *s

    result = collections.defaultdict(set)
    cluster_result = collections.defaultdict(set)
    class_id, cid = {}, 1
    categories, prefixes = {}, set()
    number = re.compile('\d+')
    with open(filename) as fin:
        start_cat, start_word = False, False
        for line in fin:
            line = line.strip()
            if start_cat and line == '%':
                start_word = True
                continue
            if line == '%':
                start_cat = True
                continue
            if start_cat and not start_word:
                parts = line.split()
                categories[int(parts[0])] = parts[1]
                continue
            if not start_word:
                continue
            parts = line.split()
            w = parts[0]
            if w.endswith('*'):
                prefixes.add(w)
            for c in parts[1:]:
                cs = re.findall(number, c)
                for n in cs:
                    n = int(n)
                    cluster_result[n].add(w)
                    result[w].add(n)
                    if n not in class_id:
                        class_id[n] = cid
                        cid += 1
    category_reverse = {v: k for (k ,v) in categories.items()}
    # return result, class_id, cluster_result, categories, category_reverse
    return {k: cluster_result[category_reverse[k]] for k in category_reverse}

def in_liwc_cluster(w, cluster=None):
    if w in cluster:
        return True
    for i in range(1, len(w)):
        k = '%s*' % w[:i]
        if k in cluster:
            return True
    return False

def count_liwc_words(text, cluster=None):
    words = text.lower().split()
    count = 0
    for word in words:
        if in_liwc_cluster(word, cluster):
            count += 1
    return count
