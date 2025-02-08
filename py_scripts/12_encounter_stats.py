import pandas as pd
import scipy.stats as stats
import pyreadr

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

        
    df.sort_values(['enc_md_id', 'start_datetime'], inplace=True)
    df['patient_count'] = 1
    count_df = df.groupby(["enc_md_id", "start_datetime"])["patient_count"].sum().reset_index()
    df.pop('patient_count')
    count_df = count_df.reset_index()

    dedup_df = df.drop_duplicates(['enc_md_id', 'start_datetime'])
    dedup_df["start_datetime"] = pd.to_datetime(dedup_df["start_datetime"])
    dedup_df = dedup_df.merge(count_df, on=["enc_md_id", "start_datetime"])
    return dedup_df, top_cc, cc



if __name__ == "__main__":
    dedup_df, top_cc, cc = load_data()

    controlled_TIME = "+".join([f"C({c})" for c in ["day_of_week", "week_of_year", "year"]]) # no time of day since we are measuring doctor on the day level
    controlled_MD = "C(enc_md_id)"

    import statsmodels.formula.api as smf
    controls = f"{controlled_TIME} + {controlled_MD}"
    model = smf.ols(f"patient_count ~  days_worked_past_week + {controls}", data=dedup_df)
    result = model.fit()