import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
sys.path.append("./py_scripts/")

all_cohort_note = pd.read_csv("./data/processed_notes.csv")

train_cohort_note = all_cohort_note[all_cohort_note.split=="train"]
test_cohort_note = all_cohort_note[all_cohort_note.split=="test"]
cohort_note = pd.concat([train_cohort_note, test_cohort_note], axis=0).reset_index()

print("split train/validation")
train = cohort_note[cohort_note.split=="train"]
train, validation = train_test_split(train, test_size=0.2)
test = cohort_note[cohort_note.split=="test"]

print("saving langauge model datasets")
Path("./data/lm/").mkdir(parents=True, exist_ok=True)
train.to_csv("./data/lm/train.csv", index=False)
validation.to_csv("./data/lm/validation.csv", index=False)
test.to_csv("./data/lm/test.csv", index=False)

print("Finished!")