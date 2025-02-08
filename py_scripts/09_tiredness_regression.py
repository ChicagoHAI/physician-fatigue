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
    
    day_night_shift = pd.read_csv(f"{DATA_ROOT}/day_night_shift.csv")
    df = df.merge(day_night_shift, on="ed_enc_id")
    
    patient_cnt_df = pd.read_csv(f"{DATA_ROOT}/patient_cnt_shift.csv")
    df = df.merge(patient_cnt_df, on="ed_enc_id")
    
    start_var_df = pd.read_csv(f"{DATA_ROOT}/start_time_var_shift.csh")
    df = df.merge(start_var_df[["ed_enc_id", "start_hour_var"]], on="ed_enc_id")
    df["start_hour_var"] = df["start_hour_var"].fillna(0)
    
    
    print(df.shape, df.columns.values)
    
    print(df["days_worked_past_week"].agg(["mean", "sem"]))
    print(df["patient_cnt_shift"].agg(["mean", "sem"]))
    print(df.groupby("split")["ppl_log"].describe())
    return df, top_cc, cc

def get_tiredness_coef_pvalue(results):
    table = results.summary().tables[1]
    results_as_html = table.as_html()
    table_df = pd.read_html(results_as_html, header=0, index_col=0)[0]#.reset_index()
    coef, std_err, pvalue = table_df.loc["tiredness_score"][["coef", "std err", "P>|t|"]].values
    if pvalue < 0.001:
        coef = str(coef)+"***"
    elif pvalue < 0.01:
        coef = str(coef)+"**"
    elif pvalue < 0.05:
        coef = str(coef)+"*"
    else:
        coef = str(coef)
    return coef, std_err, pvalue

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

def get_coef_pvalue(results, coef_name):
    table = results.summary().tables[1]
    results_as_html = table.as_html()
    table_df = pd.read_html(results_as_html, header=0, index_col=0)[0]#.reset_index()
    coef, std_err, pvalue = table_df.loc[coef_name][["coef", "std err", "P>|t|"]].values
    if pvalue < 0.001:
        coef = str(coef)+"***"
    elif pvalue < 0.01:
        coef = str(coef)+"**"
    elif pvalue < 0.05:
        coef = str(coef)+"*"
    else:
        coef = str(coef)
    return coef, std_err

def regression(df, formula, name, regression_output, regression_intercept_output):
    results = smf.ols(
        formula, 
        data=df
    ).fit()
    
    # print("=============== regression for: ", name, "==================\n\n")
    # print(results.summary())
    with open(f"/PHShome/cw203/{OUTPUT_DIR}/{name}.txt", "w") as f:
        f.write(str(results.summary()))
    
    if "tiredness_score" in formula:
        coef, std_err, pvalue = get_tiredness_coef_pvalue(results)
        regression_output.append({
            "dependent_var": name,
            "coef" : coef,
            "stderr" : std_err,
            "pvalue" : pvalue
        })
    elif "days_worked_past_week" in formula:
        coef, std_err, pvalue = get_workload_coef_pvalue(results)
        regression_output.append({
            "dependent_var": name,
            "coef" : coef,
            "stderr" : std_err,
            "pvalue" : pvalue
        })
    coef, std_err = get_intercept_coef_pvalue(results)
    regression_intercept_output.append({
        "dependent_var": name,
        "coef" : coef,
        "stderr" : std_err
    })


if __name__ == "__main__":
    
    df, top_cc, cc = load_data()

    controlled_CC = "+".join([f"C({c})" for c in cc])
    controlled_TIME = "+".join([f"C({c})" for c in ["time_of_day", "day_of_week", "week_of_year", "year"]])
    controlled_TIME_except_ToD = "+".join([f"C({c})" for c in ["day_of_week", "week_of_year", "year"]])
    controlled_DEM = "C(patient_race) + C(patient_sex) + dem_age"
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

    # days worked
    regression(
        cc_tired_df,
        f"days_worked_past_week ~  tiredness_score + {controlled_CC} + {controlled_DEM} + {controlled_MD} + {controlled_TIME} + {controlled_ENCOUNTER}",
        "days_worked_past_week",
        regression_output,
        regression_intercept_output
    )

    # patient_cnt_shift
    regression(
        cc_tired_df,
        f"patient_cnt_shift ~  tiredness_score + {controlled_CC} + {controlled_DEM} + {controlled_MD} + {controlled_TIME} + {controlled_ENCOUNTER}",
        "patient_cnt_shift",
        regression_output,
        regression_intercept_output
    )
    
    # night vs day [1-5] vs [6-23]
    cc_tired_df["is_night"] = (cc_tired_df.time_of_day.isin([1,2,3,4,5])).astype(int)
    # drop time of day
    regression(
        cc_tired_df,
        f"is_night ~  tiredness_score + {controlled_CC} + {controlled_DEM} + {controlled_MD} + {controlled_TIME_except_ToD} + {controlled_ENCOUNTER}",
        "is_night",
        regression_output,
        regression_intercept_output
    )
    
    # Shift variance
    regression(
        cc_tired_df,
        f"start_hour_var ~ tiredness_score  + {controlled_CC} + {controlled_DEM} + {controlled_MD} + {controlled_TIME} + {controlled_ENCOUNTER}",
        "start_hour_var",
        regression_output,
        regression_intercept_output
    )


    # precision of heart attach testing p__ensemble__stent_or_cabg_010_day__tested
    cc_tired_df["stent_or_cabg_010_day"] = cc_tired_df["stent_or_cabg_010_day"].astype(float)
    cc_tired_df["test_010_day"] = cc_tired_df["test_010_day"].astype(float)
    
    regression(
        cc_tired_df[cc_tired_df.test_010_day==1],
        f"stent_or_cabg_010_day ~  tiredness_score  + {controlled_CC} + {controlled_DEM} + {controlled_MD} + {controlled_TIME} + {controlled_ENCOUNTER}",
        "yield_tiredness_all_controlled",
        regression_output,
        regression_intercept_output
    )
    regression(
        cc_tired_df[cc_tired_df.test_010_day==1],
        f"stent_or_cabg_010_day ~  days_worked_past_week  + {controlled_CC} + {controlled_DEM} + {controlled_MD} + {controlled_TIME} + {controlled_ENCOUNTER}",
        "yield_workload_all_controlled",
        regression_output,
        regression_intercept_output
    )
    
    
    tested_df = cc_tired_df[cc_tired_df.test_010_day==1]

    md_count_df = tested_df.groupby("enc_md_id").size().sort_values(ascending=False)
    note_md_tested = md_count_df[md_count_df>=20].index
    tested_df["frequen_md_id"] = tested_df["enc_md_id"].apply(lambda x: x if x in note_md_tested else "other")
    regression(
            tested_df,
            f"stent_or_cabg_010_day ~  tiredness_score  + {controlled_CC} + {controlled_DEM}  + {controlled_TIME} + C(frequen_md_id) + {controlled_ENCOUNTER}",
            "yield_tiredness_frequentMD_20notes",
            regression_output,
            regression_intercept_output
    )
    regression(
        tested_df,
        f"stent_or_cabg_010_day ~  days_worked_past_week  + {controlled_CC}+ {controlled_MD} + {controlled_TIME} + C(frequen_md_id) + {controlled_ENCOUNTER}",
        "yield_workload_frequentMD_20notes",
        regression_output,
        regression_intercept_output
    )

    # print output and intercept resutls group by dependent_var
    output = []
    for coef, intercept in zip(regression_output, regression_intercept_output):
        output.append(
            {
                "dependent_var": coef["dependent_var"],
                "pvalue": coef["pvalue"],
                "coef_value": coef["coef"],
                "coef_stderr": coef["stderr"],
                "intercept_value": intercept["coef"],
                "intercept_stderr": intercept["stderr"],
            }
        )
    output_df = pd.DataFrame(output)
    output_df.to_csv(f"./data/regression_results/regression_results_review_reg.csv", index=False)
    