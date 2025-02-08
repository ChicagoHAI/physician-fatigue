import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
sns.set_style('white')
sns.set_context("paper", font_scale = 2)

DATA_ROOT = "./data"


def load_data():

    test_df = pd.read_csv("./data/test.csv")
    model = "lr_cla"

    target = "days_worked_past_week"
    tiredness_df = pd.read_csv(f"{DATA_ROOT}/tiredness_exp/test_tiredness_score_structured_{target}:NPSM:LIWC:Read:CC:balanced_content_sections_{model}.csv")
    test_df = test_df.merge(tiredness_df, on="ed_enc_id")
    
    print(test_df.shape, test_df.columns.values)
    
    return test_df


if __name__ == "__main__":

    cc_tired_df = load_data()

    tiredness_std = cc_tired_df.tiredness_score.std()
    tiredness_mean = cc_tired_df.tiredness_score.mean()

    

    plt.rcParams['figure.figsize']=(8,5)
    shift_df = cc_tired_df.copy()
    shift_df["tiredness_score"] = (shift_df.tiredness_score - tiredness_mean)/tiredness_std
    
    shift_df.time_of_day = (shift_df.time_of_day-1)%24
    sns.despine()
    sns.set_style("white")
    fig, ax = plt.subplots(1,1)
    colors = sns.color_palette('muted')
    p1 = colors.as_hex()[0]
    p2 = colors.as_hex()[1]

    ax= sns.lineplot(data=shift_df[(shift_df.time_of_day.isin([0,1,2,3,4]))], err_style="bars", errorbar="se",
        x="time_of_day", y=f"tiredness_score", ax=ax, linestyle='None',color=p1)

    ax= sns.lineplot(data=shift_df[~(shift_df.time_of_day.isin([0,1,2,3,4]))], err_style="bars", errorbar="se",
        x="time_of_day", y=f"tiredness_score", ax=ax, linestyle='None', color=p2)

    sns.despine()


    ax= sns.regplot(data=shift_df[(shift_df.time_of_day.isin([0,1,2,3,4]))], 
        x="time_of_day", y=f"tiredness_score", ax=ax, scatter=False, ci=None,color=p1, label="overnight shifts")

    ax= sns.regplot(data=shift_df[~(shift_df.time_of_day.isin([0,1,2,3,4]))], 
        x="time_of_day", y=f"tiredness_score", ax=ax, scatter=False, ci=None,  color=p2,label="all other shifts", line_kws={"ls":"--"})
    ax.legend()

    means = shift_df[(shift_df.time_of_day.isin([0,1,2,3,4]))].groupby('time_of_day')['tiredness_score'].mean().values
    plt.scatter(y=means, x=np.arange(5), color=p1, marker='D')

    means = shift_df[~(shift_df.time_of_day.isin([0,1,2,3,4]))].groupby('time_of_day')['tiredness_score'].mean().values
    plt.scatter(y=means, x=np.arange(5,24), color=p2, marker='D')


    ax.set_ylabel("Predicted fatigue (SD units)")
    ax.set_xlabel("Patient arrival time")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticklabels(["", "", "1"]+[""]*3+["5","6",""]+[""]*16+["24"]+[""])
    fig.savefig(f"./data/tiredness_figs/tiredness_time_of_day_1_5_std_unit_reg.pdf", bbox_inches='tight')