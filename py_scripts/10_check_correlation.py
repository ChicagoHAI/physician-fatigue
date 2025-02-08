import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import pyreadr
import json
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

    md_workload = pd.read_csv(f"{DATA_ROOT}/workload.csv")
    df = df.merge(md_workload, on="ed_enc_id")
    
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

def regression(
        df,
        formula,
        depepdent_var,
        independent_var,
        regression_output,
        regression_intercept_output):
    results = smf.ols(
        formula, 
        data=df
    ).fit()
    
    coef, std_err = get_coef_pvalue(results, independent_var)
    regression_output.append({
        "dependent_var": depepdent_var,
        "independent_var": independent_var,
        "coef" : coef,
        "stderr" : std_err
    })
    coef, std_err = get_coef_pvalue(results, "Intercept")
    regression_intercept_output.append({
        "dependent_var": depepdent_var,
        "independent_var": independent_var,
        "coef" : coef,
        "stderr" : std_err
    })

def run_CC_regression(
        df,
        cc,
        controls,
        independent_var,):
    print(f"Run CC regressions with {independent_var}")
    print(f"Df length", len(df))
    
    cc_cor_dfs = []
    for c in cc:
        results = smf.ols(
            f"{c} ~ {independent_var} + {controls}", 
            data=df
        ).fit()
        results_as_html = results.summary().tables[1].as_html()
        coef_all = pd.read_html(results_as_html, header=0, index_col=0)[0].reset_index()
        input_features = [independent_var]
        res_df = coef_all[coef_all["index"].isin(input_features)]
        res_df["cc"] = c 
        cc_cor_dfs.append(res_df)
        
    cc_coef_df = pd.concat(cc_cor_dfs).sort_values("P>|t|")
    # count percentage of significant CC
    print(f"{independent_var} significant CC and total: ", len(cc_coef_df[cc_coef_df["P>|t|"]<0.05]), len(cc_coef_df))
    print(f"{independent_var} significant CC percentage: ", len(cc_coef_df[cc_coef_df["P>|t|"]<0.05])/len(cc_coef_df))

    c_count = []
    for c in cc:
        c_count.append(
            {
                "cc":c,
                "count":len(df[df[c]==1])
            }
        )
    cc_count_df = pd.DataFrame(c_count)
    cc_coef_df.merge(cc_count_df, on="cc")
    cc_coef_df.to_csv(f"/PHShome/cw203/sanity_check_tables/sanity_check_cc_{independent_var}_coefs.csv")
    
    fig, ax = plt.subplots(1)
    ax = sns.lineplot(cc_coef_df.merge(cc_count_df, on="cc")["P>|t|"])
    ax.set_ylabel("p-value")
    ax.set_xlabel("Chief Complaint Order")
    # ax.set_title("Chief complaints sorted by p-values")
    ax.axhline(0.05, color="grey")
    sns.despine()
    fig.savefig(f"./data/sanitycheck_figs/sanity_check_cc_{independent_var}_pvalues.pdf", bbox_inches='tight')

    



if __name__ == "__main__":
    
    df, top_cc, cc = load_data()

    df["is_night"] = (df.time_of_day.isin([1,2,3,4,5])).astype(int)
    df["stent_or_cabg_010_day"] = df["stent_or_cabg_010_day"].astype(float)
    df["test_010_day"] = df["test_010_day"].astype(float)
    df["is_female"] = (df["patient_sex"] == "female" ).astype(int)
    df["is_white"] = (df["patient_race"] == "white").astype(int)
    
    dependent_var_list = ["is_female", "is_white", "dem_age"]
    
    independent_var_list = [
        "days_worked_past_week_cla",
    ]

    controlled_CC = "+".join([f"C({c})" for c in cc])
    controlled_TIME = "+".join([f"C({c})" for c in ["time_of_day", "day_of_week", "week_of_year", "year"]])
    controlled_TIME_except_ToD = "+".join([f"C({c})" for c in ["day_of_week", "week_of_year", "year"]])
    controlled_DEM = "C(patient_race) + C(patient_sex) + dem_age"
    controlled_MD = "C(enc_md_id)"


    regression_output = []
    regression_intercept_output = []
    
    # Check demographic variables
    for independent_var in independent_var_list:
        for dependent_var in dependent_var_list:    

            if independent_var == "is_night":
                controls = controlled_CC + "+" + controlled_TIME_except_ToD + "+" + controlled_MD
            else:
                controls = controlled_CC + "+" + controlled_TIME + "+" + controlled_MD
            
            if independent_var == "stent_or_cabg_010_day":
                tmp_df = df[df["test_010_day"] == 1]
            elif independent_var == "days_worked_past_week_cla":
                tmp_df = df[df.days_worked_past_week.isin([1,5,6,7,])]
                tmp_df[independent_var] = df["days_worked_past_week"].apply(lambda x: 1 if x >= 5 else 0).astype(int)
            else:
                tmp_df = df

            regression(
                tmp_df, 
                f"{dependent_var} ~ {independent_var} + {controls}", 
                dependent_var,
                independent_var,
                regression_output, 
                regression_intercept_output
            )
    # print output and intercept resutls group by dependent_var
    output = []
    for coef, intercept in zip(regression_output, regression_intercept_output):
        output.append(
            {
                "dependent_var": coef["dependent_var"],
                "independent_var": coef["independent_var"],
                "coef_value": coef["coef"],
                "coef_stderr": coef["stderr"],
                "intercept_value": intercept["coef"],
                "intercept_stderr": intercept["stderr"],
            }
        )
    
    # Check CC variables
    for independent_var in independent_var_list:
        
        if independent_var == "is_night":
                controls = controlled_DEM + "+" + controlled_TIME_except_ToD + "+" + controlled_MD
        else:
            controls = controlled_DEM + "+" + controlled_TIME + "+" + controlled_MD
        
        if independent_var == "stent_or_cabg_010_day":
            tmp_df = df[df["test_010_day"] == 1]
        elif independent_var == "days_worked_past_week_cla":
            tmp_df = df[df.days_worked_past_week.isin([1,5,6,7,])]
            tmp_df[independent_var] = df["days_worked_past_week"].apply(lambda x: 1 if x >= 5 else 0).astype(int)
        else:
            tmp_df = df
        run_CC_regression(
            tmp_df, 
            cc,
            controls,
            independent_var,
        )
    
