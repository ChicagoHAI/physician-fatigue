import pandas as pd
from collections import Counter

def get_empi_ptid_dic():
    xwalk = pd.read_feather("./data/id_mapping.feather")
    empi2ptid = {}
    ptid2empi = {}
    for empi, ptid in zip(xwalk['empi'].values, xwalk['ptid'].values):
        empi2ptid[empi] = ptid
        ptid2empi[ptid] = empi
    return empi2ptid, ptid2empi

empi2ptid, ptid2empi = get_empi_ptid_dic()

note_scource = ["bwh_ed_100k", "icmp", "pe"]
notes = []
for note_name in note_scource:
    df = pd.read_csv("./data/rpdr_lno_master_{}.csv".format(note_name))
    if "chunkid" in df.columns:
        df = df.drop(columns=["chunkid"])
    notes.append(df)
note = pd.concat(notes, axis=0)

note['lno_date'] = pd.to_datetime(note['lno_date'])
sorted_note_by_date = note.sort_values(by=['empi', 'lno_date'])


# Load test/dev/test cohort data
cohorts = []
splits = ['train', 'val', 'test']

for split in splits:
    cohorts.append(pd.read_csv(f"./data/{split}_cohort.csv"))

cohort_df = pd.concat(cohorts)
cohort_df = cohort_df.sort_values(by=['ptid', 'start_datetime'])
merge_df = cohort_df.merge(sorted_note_by_date, on="ptid")
clean_df = merge_df

# only keep physician notes
ed_with_edvisit = clean_df[clean_df.comments.str.contains(r"^EDVISIT")].ed_enc_id.unique()
enc_with_edvisit_df = clean_df[clean_df.comments.str.contains(r"^EDVISIT")]

md_names = enc_with_edvisit_df.groupby("enc_md_id")["author"].agg(lambda x: Counter(x).most_common(1)[0]).to_frame()
md_names["MD_name"] = md_names.author.apply(lambda x: x[0])
md_names["MD_name_count"] = md_names.author.apply(lambda x: x[1])
md_names = md_names.sort_values(by=["MD_name", "MD_name_count"], ascending=[True,False])
md_names = md_names.drop_duplicates(subset=["MD_name"]).reset_index()
md_names = md_names[["enc_md_id", "MD_name"]].rename(columns={"MD_name":"author"})

ed_with_edvisit_cleaned_md_df = enc_with_edvisit_df.merge(md_names, on=["enc_md_id", "author"])
ed_with_edvisit_cleaned_md_df["note_len"] = ed_with_edvisit_cleaned_md_df.comments.apply(len)
ed_with_edvisit_cleaned_md_df = ed_with_edvisit_cleaned_md_df.sort_values(
    ["ed_enc_id","note_len"], ascending=[True, False]).drop_duplicates(subset=["ed_enc_id"])

ed_with_edvisit_cleaned_md_df.to_csv("./data/note_match_EDVISIT_cleaned_MD.csv", index=False)
