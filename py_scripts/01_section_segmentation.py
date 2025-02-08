import pandas as pd
import utils

df = pd.read_csv("./data/note_match_EDVISIT_cleaned_MD.csv")
section_set = utils.get_note_sections(df, "comments")
section_df = utils.segment_note2sections(df, section_set, "comments")

des_df = section_df.describe().T.sort_values("count", ascending=False)
des_df['ratio'] = des_df['count'].astype(float)/len(section_df)
print(des_df.round(2).drop(columns=["top"]).to_markdown())

segmented_df = pd.concat([df, section_df], axis=1)
segmented_df.to_csv("./data/section_note_match_EDVISIT_cleaned_MD.csv", index=False)
