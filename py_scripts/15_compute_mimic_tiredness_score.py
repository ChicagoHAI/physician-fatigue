import pickle
import pandas as pd
import pyreadr 
import numpy as np
from scipy.sparse import csr_matrix, hstack

# Load the best model and vectorizer
output_path = "./data/tiredness_exp/"
model_type = "lr_cla"
feature_used = "structured"
section = "content_sections"
task = "days_worked_past_week:NPSM:LIWC:Read:CC:balanced"

best_model = pickle.load(
    open(
        f"{output_path}/{model_type}/{task}_{feature_used}_{section}_{model_type}_predictor.pkl",
        "rb",
    )
)


scaler = pickle.load(
    open(
        f"{output_path}/{model_type}/{task}_{feature_used}_{section}_{model_type}_scaler.pkl",
        "rb",
    )
)
imputer = pickle.load(
    open(
        f"{output_path}/{model_type}/{task}_{feature_used}_{section}_{model_type}_imputer.pkl",
        "rb",
    )
)

feature_list = [
    "note_len", "ppl_log",
    "stopword_frac", "medicalword_frac"
] #
liwc_categories = [
'pronoun_frac', 'i_frac', 'we_frac','you_frac','shehe_frac','they_frac','ipron_frac',
'affect_frac','posemo_frac','negemo_frac','anx_frac','anger_frac','sad_frac',
'cogmech_frac','insight_frac','cause_frac','discrep_frac','tentat_frac','certain_frac','inhib_frac','incl_frac','excl_frac']


readability_features = [
    "readability_flesch_kincaid"
    ]

input_df = pd.read_csv("./data/mimic_phy_processed.csv")
result = pyreadr.read_r('./data/cc_full_cohort.rds')[None]
cc = [s for s in result.columns if s.startswith("cc_")]

# add cc to input_df as columns and make all value 0
input_df = pd.concat([input_df, pd.DataFrame(0, index=input_df.index, columns=cc)], axis=1)

cols = feature_list + liwc_categories + readability_features

test_features = []

test_standardized_X = imputer.transform(
    input_df[cols]
)
test_standardized_X = scaler.transform(test_standardized_X)

test_standardized_X = csr_matrix(test_standardized_X)

test_features.append(csr_matrix(test_standardized_X))

test_features.append(csr_matrix(input_df[cc].values))

test_input_X = hstack(test_features)

input_df["tiredness_score"] = best_model.predict_proba(test_input_X)[:, 1]

input_df.to_csv("./data/mimic_phy_processed_tiredness.csv")