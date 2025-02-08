import pandas as pd
import statsmodels.formula.api as smf

def load_data():
    df = pd.read_csv("./data/mimic_phy_processed_tiredness.csv")

    if "GENDER" in df.columns:
        df.pop("GENDER")
    df.pop("SUBJECT_ID")
    df['CHARTTIME'] = pd.to_datetime(df['CHARTTIME'], errors='coerce')
    df['STORETIME'] = pd.to_datetime(df['STORETIME'], errors='coerce')
    df = df.dropna(subset=['CHARTTIME'])
    admission_df = pd.read_csv("/scratch/c/cw203/mimic_admission.csv")
    # for ETHNICITY, only keep WHITE, BLACK, HISPANIC and put others into "OTHER"
    admission_df["ETHNICITY"] = admission_df["ETHNICITY"].apply(lambda x: x if x in ["WHITE", "BLACK", "HISPANIC"] else "OTHER")


    admission_df['ADMITTIME_DETAIL'] = pd.to_datetime(admission_df['ADMITTIME'], errors='coerce')

    # # use "ADMITTIME" and "DOB" to compute age
    admission_df['ADMITTIME'] = pd.to_datetime(admission_df['ADMITTIME']).dt.date
    admission_df['DOB'] = pd.to_datetime(admission_df['DOB']).dt.date
    admission_df['AGE'] = admission_df.apply(lambda x: (x['ADMITTIME'] - x['DOB']).days // 365, axis=1)

    # # use "ADMITTIME" and "DISCHTIME" to compute length of stay in day
    admission_df['DISCHTIME'] = pd.to_datetime(admission_df['DISCHTIME'], errors='coerce').dt.date
    admission_df['LOS'] = (admission_df['DISCHTIME'] - admission_df['ADMITTIME']).dt.days

    df = df.merge(admission_df, on="HADM_ID")
    df["admit_time_of_day"] = df["ADMITTIME_DETAIL"].dt.hour
    df["hour_since_admission"] = (df["CHARTTIME"] - df["ADMITTIME_DETAIL"]).dt.total_seconds() / 3600
    return df


df = load_data()
admit_df = df[df["DESCRIPTION"].str.contains("admission note", case=False)]
md_count_df = admit_df.groupby("writer_id").size().sort_values(ascending=False)
note_md_tested = md_count_df[md_count_df>=20].index
admit_df["frequen_md_id"] = admit_df["writer_id"].apply(lambda x: x if x in note_md_tested else "other")
admit_df["is_night"] = admit_df["admit_time_of_day"].isin([1,2,3,4,5])
admit_df["is_night"] = admit_df["is_night"].astype(float)

results = smf.ols(
        "is_night ~ tiredness_score + C(ETHNICITY, Treatment(reference='WHITE')) + C(GENDER) +  C(INSURANCE) + AGE + LOS + C(DIAGNOSIS) + C(frequen_md_id)", #
        data=admit_df[admit_df["hour_since_admission"]<=6]
    ).fit()
results.summary()
