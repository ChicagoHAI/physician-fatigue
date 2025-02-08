import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import pyreadr
import statsmodels.formula.api as smf

DATA_ROOT = "./data"
TIREDNESS_DATA_PATH = f"{DATA_ROOT}/tiredness_exp/data/"
OUTPUT_DIR = "tiredness_regression_review"

def load_data():
    train_df = pd.read_csv(f"{TIREDNESS_DATA_PATH}/train.csv")
    test_df = pd.read_csv(f"{TIREDNESS_DATA_PATH}/test.csv")

    df = pd.concat([train_df, test_df])

    categorical_cols = ['black', 'hispanic', 'white', 'other'] 
    race2label_dict = {r:i for i, r in enumerate(categorical_cols)}
    label2race_dict = {i:r for i, r in enumerate(categorical_cols)}
    df["patient_race"] = df["dem_race_label"].apply(lambda x: label2race_dict[x])
    df["patient_sex"] = df["dem_sex_female"].apply(lambda x: "female" if x else "male")
    
    
    
    result = pyreadr.read_r(f'{DATA_ROOT}/cc_full_cohort.rds')[None]
    cc = [s for s in result.columns if s.startswith("cc_")]
    df = df.merge(result, on="ed_enc_id")
    cc = [s for s in result.columns if s.startswith("cc_")]
    out = []
    for c in cc:
        out.append({"chief_complaint":c, "count": result[c].sum()})
    top_cc = pd.DataFrame(out).sort_values("count", ascending=False).iloc[:50]

    # add encounter conditions
    cond_df = pd.read_csv(f"{DATA_ROOT}/encounter_conditions.csv")
    df = df.merge(cond_df, on="ed_enc_id")

    md_workload = pd.read_csv(f"{DATA_ROOT}/workload.csv")
    df = df.merge(md_workload, on="ed_enc_id")
    
    return df, top_cc, cc

def get_workload_coef_pvalue(results):
    table = results.summary().tables[1]
    results_as_html = table.as_html()
    table_df = pd.read_html(results_as_html, header=0, index_col=0)[0]#.reset_index()
    coef, std_err, pvalue = table_df.loc["days_worked_past_week"][["coef", "std err", "P>|t|"]].values
    if pvalue < 0.001:
        coef = str(coef)+"***"
    elif pvalue < 0.01:
        coef = str(coef)+"**"
    elif pvalue < 0.05:
        coef = str(coef)+"*"
    else:
        coef = str(coef)
    return coef, std_err, pvalue

def get_intercept_coef_pvalue(results):
    table = results.summary().tables[1]
    results_as_html = table.as_html()
    table_df = pd.read_html(results_as_html, header=0, index_col=0)[0]#.reset_index()
    coef, std_err, pvalue = table_df.loc["Intercept"][["coef", "std err", "P>|t|"]].values
    if pvalue < 0.001:
        coef = str(coef)+"***"
    elif pvalue < 0.01:
        coef = str(coef)+"**"
    elif pvalue < 0.05:
        coef = str(coef)+"*"
    else:
        coef = str(coef)
    return coef, std_err


df, top_cc, cc = load_data()
controlled_CC = "+".join([f"C({c})" for c in cc])

results = smf.ols(
    f"los_days ~ days_worked_past_week + C(enc_md_id) + C(time_of_day) + C(day_of_week) + C(week_of_year) + C(year)", 
    data=df
).fit()
print(results.summary())