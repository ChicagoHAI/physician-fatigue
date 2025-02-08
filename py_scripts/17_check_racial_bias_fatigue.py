import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import pyreadr
import statsmodels.formula.api as smf

DATA_ROOT = "./data"
TIREDNESS_DATA_PATH = f"{DATA_ROOT}/tiredness_exp/data/"


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
    
    print(df.shape, df.columns.values)
    
    print(df["days_worked_past_week"].agg(["mean", "sem"]))
    print(df["patient_cnt_shift"].agg(["mean", "sem"]))
    print(df.groupby("split")["ppl_log"].describe())
    return df, top_cc, cc


if __name__ == "__main__":
    df, top_cc, cc = load_data()
    controlled_CC = "+".join([f"C({c})" for c in cc])

    controlled_TIME = "+".join([f"C({c})" for c in ["time_of_day", "day_of_week", "week_of_year", "year"]])
    controlled_TIME_except_ToD = "+".join([f"C({c})" for c in ["day_of_week", "week_of_year", "year"]])
    controlled_DEM = "C(patient_race, Treatment(reference='white'))"
    controlled_MD = "C(enc_md_id)"
    controlled_ENCOUNTER = "C(InsuranceClass) + los_days"

    regression_output = []
    regression_intercept_output = []
    section = "content_sections"
    feature_used = "structured"
    model = "lr_cla"
    target = "days_worked_past_week"
    tiredness_df = pd.read_csv(f"{DATA_ROOT}/tiredness_exp/test_tiredness_score_structured_{target}:NPSM:LIWC:Read:CC:balanced_content_sections_{model}.csv")

    cc_tired_df = df.merge(tiredness_df, on="ed_enc_id")

    cc_tired_df["is_night"] = cc_tired_df["time_of_day"].apply(lambda x: 1 if x in [1,2,3,4,5] else 0)

    results = smf.ols(
        f"days_worked_past_week ~ {controlled_CC} + {controlled_TIME} + {controlled_DEM} + {controlled_MD} + {controlled_ENCOUNTER}",
        data=cc_tired_df
    ).fit()

    results = smf.ols(
        f"tiredness_score ~ {controlled_CC} + {controlled_TIME} + {controlled_DEM} + {controlled_MD} + {controlled_ENCOUNTER}",
        data=cc_tired_df
    ).fit()

    results = smf.ols(
        f"tiredness_score ~ {controlled_CC} + {controlled_TIME_except_ToD} + {controlled_DEM} + {controlled_MD} + {controlled_ENCOUNTER} + C(is_night)",
        data=cc_tired_df
    ).fit()