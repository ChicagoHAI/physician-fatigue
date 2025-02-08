import pandas as pd

cohorts = []
splits = ['train', 'val', 'test']

for split in splits:
    cohorts.append(pd.read_csv(f"./data/modeling/cohorts/random/{split}_cohort.csv"))
df = pd.concat(cohorts, axis=0)
orginal_df = df[["ed_enc_id", "start_datetime"]]


train_df = pd.read_csv("./data/lm/train.csv")
valid_df = pd.read_csv("./data/lm/validation.csv")
test_df = pd.read_csv("./data/lm/test.csv")
df = pd.concat([train_df, valid_df, test_df], axis=0)
df = df.drop(columns=["start_datetime"]) # drop simplified datetime
df = df.merge(orginal_df, on="ed_enc_id") # get the original dateime with hourly information
df.start_datetime = pd.to_datetime(df.start_datetime, errors='coerce')
df.sort_values(['enc_md_id', 'start_datetime'], inplace=True)

df_day_workload = df.copy()
df_day_workload["date"] = df_day_workload["start_datetime"].dt.date
df_day_workload.date = pd.to_datetime(df_day_workload.date, errors='coerce')

dedup_df_day_workload= df_day_workload.drop_duplicates(["enc_md_id", "date"]).copy()
dedup_df_day_workload.date = pd.to_datetime(dedup_df_day_workload.date, errors='coerce')

dedup_df_day_workload.set_index('date', inplace=True)
dedup_df_day_workload['days_worked_past_week'] = dedup_df_day_workload.groupby('enc_md_id')['index'].apply(lambda x: x.rolling('7D').count())
dedup_df_day_workload = dedup_df_day_workload.reset_index()

past_week_workload_df = dedup_df_day_workload[["ed_enc_id", "days_worked_past_week"]]

past_week_workload_df.to_csv("./data/workload.csv", index=False)