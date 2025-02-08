import itertools
import json
import multiprocessing as mp
import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from nltk.corpus import stopwords
import utils
from multiprocessing import Pool
import pyreadr

data_path = "./data/lm"
output_path = "./data/tiredness_exp/"
debug = False


def precision_at_k_percent(y_label, y_pred, k=10):
    pairs = list(zip(y_label, y_pred))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = pairs[: int((len(pairs) / 100) * k)]
    return sum(p[0] for p in top_pairs) / len(top_pairs)


def precision_at_k(y_label, y_pred, k=100):
    pairs = list(zip(y_label, y_pred))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = pairs[:k]
    return sum(p[0] for p in top_pairs) / k


def evaluate_with_hyperparameter(X, y, model, k_fold=5, tune=True):

    if tune:
        print("run grid search for model")
        s_kfold = StratifiedKFold(n_splits=k_fold, random_state=42, shuffle=True)
        params = {
            "C": np.logspace(-5, 1, 5),
        }
        print("run lr classification model")
        lr_model = LogisticRegression(n_jobs=32, random_state=42, max_iter=1000)
        best_model = GridSearchCV(
            lr_model,
            params,
            n_jobs=32,
            cv=s_kfold,
            scoring="roc_auc",
            verbose=3,
            refit=True,
        )
        print("===start re-fitting on all data===")
        best_model.fit(X, y)

    else:
        
        best_model = LogisticRegression(penalty="none", random_state=42)
        best_model.fit(X, y)

    train_preds = best_model.predict(X)
    print("best train acc", accuracy_score(y, train_preds))
    print(
        "best train macro f1",
        precision_recall_fscore_support(
            y, train_preds, pos_label=True, average="binary"
        ),
    )

    return best_model


def train_test_pipeline(
    train_cohort,
    train_X,
    test_cohort,
    test_X,
    feature_used,
    task,
    model_type="lr",
    section=None,
    chief_complaints=None,
    structured_cols=None
):
    assert feature_used in ["structured", "CConly"]

    train_features = []
    if feature_used in ["structured"]:

        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        train_standardized_X = imputer.fit_transform(train_X)

        scaler = StandardScaler()
        train_standardized_X = scaler.fit_transform(train_standardized_X)

        train_standardized_X = csr_matrix(train_standardized_X)
        print("get standardized features")
        print(
            "train standardized shape",
            train_standardized_X.shape,
            "type",
            type(train_standardized_X),
        )
        train_features.append(train_standardized_X)
        pickle.dump(
            scaler,
            open(
                f"{output_path}/{model_type}/{task}_{feature_used}_{section}_{model_type}_scaler.pkl",
                "wb",
            ),
        )
        pickle.dump(
            imputer,
            open(
                f"{output_path}/{model_type}/{task}_{feature_used}_{section}_{model_type}_imputer.pkl",
                "wb",
            ),
        )

    if chief_complaints:
        train_features.append(csr_matrix(train_cohort[chief_complaints].values))

    train_input_X = hstack(train_features)

    y_target = f"label"
    # train model
    y = train_cohort[y_target].astype(int).to_frame()
    print("training data shapes", train_input_X.shape, y.shape)
    model = evaluate_with_hyperparameter(train_input_X, y, model=model_type, tune=True)
    try:
        coefs = model.coef_[0]
    except:
        coefs = model.best_estimator_.coef_[0]

    feature_names = []
    if structured_cols:
        feature_names.extend(structured_cols)
    if chief_complaints:
        feature_names.extend(chief_complaints)
    coef_dict = {}

    for coef, feat in zip(coefs, feature_names):
        coef_dict[feat] = round(coef, 5)
    sorted_coefs = sorted(coef_dict.items(), key=lambda x: x[1], reverse=True)
    top_feature_coef = sorted_coefs[:50]
    bottom_feature_coef = reversed(sorted_coefs[-50:])
    top_df = pd.DataFrame(top_feature_coef, columns=["top_feature", "coef"])
    bottom_df = pd.DataFrame(
        bottom_feature_coef, columns=["bottom_feature", "coef"]
    )

    model_dir = f"{output_path}/{model_type}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    pd.concat([top_df, bottom_df], axis=1).to_csv(
        f"{output_path}/{model_type}/{task}_{feature_used}_{section}_{model_type}_top_bottom_features.csv",
        index=False,
    )
    # save mode coef_dict
    pickle.dump(
        coef_dict,
        open(
            f"{output_path}/{model_type}/{task}_{feature_used}_{section}_{model_type}_coef_dict.pkl",
            "wb",
        ),
    )

    # save model
    model_dir = f"{output_path}/{model_type}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    pickle.dump(
        model,
        open(
            f"{output_path}/{model_type}/{task}_{feature_used}_{section}_{model_type}_predictor.pkl",
            "wb",
        ),
    )

    del train_input_X
    test_features = []

    if feature_used in ["both", "structured"]:
        test_standardized_X = imputer.transform(test_X)
        test_standardized_X = scaler.transform(test_standardized_X)
        test_standardized_X = csr_matrix(test_standardized_X)
        print("get standardized features")
        print(
            "test standardized shape",
            test_standardized_X.shape,
            "type",
            type(test_standardized_X),
        )

        test_features.append(csr_matrix(test_standardized_X))

    if chief_complaints:
        test_features.append(csr_matrix(test_cohort[chief_complaints].values))

    test_input_X = hstack(test_features)

    # evaluate on test set
    print(f"Load test model")
    y = test_cohort[y_target].astype(int).to_frame()
    best_model = pickle.load(
        open(
            f"{output_path}/{model_type}/{task}_{feature_used}_{section}_{model_type}_predictor.pkl",
            "rb",
        )
    )
    if "reg" in model_type:
        preds = best_model.predict(test_input_X)
        mse = mean_squared_error(y, preds)
        r2 = r2_score(y, preds)
        print(f"mse: {mse}")
        print(f"r2: {r2}")

        results = {
            "mse": mse,
            "r2": r2,
        }
    else:
        preds = best_model.predict(test_input_X)
        probs = best_model.predict_proba(test_input_X)[:, 1]
        acc = accuracy_score(y, preds)
        PRF = precision_recall_fscore_support(y, preds, average="binary")
        confusion_m = confusion_matrix(y, preds).tolist()
        roc = roc_auc_score(y, probs)
        ap = average_precision_score(y, probs)
        p_10_p = precision_at_k_percent(y.values.flatten(), probs)
        p_100 = precision_at_k(y.values.flatten(), probs)
        print("test p at 10 percent", p_10_p)
        print("test p at 100 instances", p_100)
        print(f"model  Acc:", acc)
        print(f"model  PRF:", PRF)
        print(f"model  roc:", roc)
        print(f"model  ap:", ap)

        results = {
            "acc": acc,
            "roc": roc,
            "ap": ap,
            "p_10_p": p_10_p,
            "p_100": p_100,
            "PRF": PRF,
            "confusion_matrix": confusion_m,
        }
    json.dump(
        results,
        open(
            f"{output_path}/{model_type}/{task}_{feature_used}_{section}_{model_type}_result.json",
            "w",
        ),
    )
    print(
        f"result save to:",
        f"{output_path}/{model_type}/{task}_{feature_used}_{section}_{model_type}_result.json",
    )


def train_test_model(
    target,
    train_df,
    test_df,
    feature_used,
    section,
    model_type,
    structured_features,
    chief_complaints,
    liwc_categories,
    readability_features,
    balanced,
    overwrite=False,
    bootstrap_id=None,
):


    task = target

    if structured_features:
        task += ":" + "".join(s[0].upper() for s in structured_features)
    if liwc_categories:
        task += ":LIWC" 
    if readability_features:
        task += ":Read"
    if chief_complaints:
        task += ":CC"

    if balanced:
        task += ":balanced"

    print("=" * 20)
    if not overwrite and os.path.exists(
        f"{output_path}/{model_type}/{task}_{feature_used}_{section}_{model_type}_result.json"
    ):
        print("result exist", task)
        return
    print(
        f"======================\nTraining/Testing between {task} on {section}\n======================"
    )

    train_cohort = train_df.dropna(subset=[target, section])
    test_cohort = test_df.dropna(subset=[target, section])

    train_cohort = train_cohort[train_cohort["days_worked_past_week"].isin([1,5,6,7])]
    test_cohort = test_cohort[test_cohort["days_worked_past_week"].isin([1,5,6,7])]

    train_cohort["label"] = train_cohort["days_worked_past_week"] >= 5
    test_cohort["label"] = test_cohort["days_worked_past_week"] >= 5

    def create_balanced_cohort(cohort, chief_complaints):
        # create balanced training set based on label and each of chief complaints
        pos_cc_cohort = []
        neg_cc_cohort = []

        for cc in chief_complaints:
            # print("create balanced cohort")
            pos = cohort[(cohort["label"] == True) & (cohort[cc] == 1)]
            neg = cohort[(cohort["label"] == False) & (cohort[cc] == 1)]
            if  len(pos) == 0 or len(neg) == 0:
                continue
            if len(pos) > len(neg):
                pos = pos.sample(neg.shape[0], random_state=42)
            else:
                neg = neg.sample(pos.shape[0], random_state=42)
            pos_cc_cohort.append(pos)
            neg_cc_cohort.append(neg)
        cohort = pd.concat(pos_cc_cohort + neg_cc_cohort)
        return cohort

    if balanced:
        train_cohort = create_balanced_cohort(train_cohort, chief_complaints)
        test_cohort = create_balanced_cohort(test_cohort, chief_complaints)


    print(f"training size: {len(train_cohort)}")
    print(train_cohort["label"].describe())
    print(f"test size: {len(test_cohort)}")
    print(test_cohort["label"].describe())
    print("structured features", structured_features)

    structured_cols = []
    if structured_features:
        structured_cols += structured_features
    if liwc_categories:
        structured_cols += liwc_categories
    if readability_features:
        structured_cols += readability_features

    train_test_pipeline(
        train_cohort,
        train_cohort[
            structured_cols
            ] if structured_features else None,
        test_cohort,
        test_cohort[
            structured_cols
            ] if structured_features else None,
        feature_used=feature_used,
        task=task,
        section=section,
        chief_complaints=chief_complaints,
        structured_cols=structured_cols,
        model_type=model_type,
    )

def run(
    train_df,
    test_df,
    section_list,
    do_train,
    do_predict,
    model_type,
    feature_used,
    structured_features,
    chief_complaints,
    liwc_categories,
    readability_features,
    balanced,
):

    print(f"Total data size. Training: {len(train_df)}, test: {len(test_df)}")

    print("=" * 20)
    print("tiredness prediction")

    if do_train:
         for section in section_list:
            print(f"section: {section}")

            cols = ["ed_enc_id", section, "time_of_day", "days_worked_past_week"]

            if structured_features:
                cols += structured_features

            if chief_complaints:
                cols += chief_complaints
            
            if liwc_categories:
                cols += liwc_categories
            
            if readability_features:
                cols += readability_features

            train_section_df = train_df[cols].dropna()
            test_section_df = test_df[cols].dropna()
            train_section_df[section] = train_section_df[section].astype(str)
            test_section_df[section] = test_section_df[section].astype(str)

            pool = mp.Pool(mp.cpu_count())
            results = pool.starmap(
                train_test_model,
                [
                    (
                        target,
                        train_section_df,
                        test_section_df,
                        feature_used,
                        section,
                        model_type,
                        structured_features,
                        chief_complaints,
                        liwc_categories,
                        readability_features,
                        balanced,
                        True,
                    )
                    for target in [
                        "days_worked_past_week",
                        ]
                ],
            )
            pool.close()
            pool.join()
    if do_predict:
        print("tiredness prediciton")
        for section in section_list:

            test_section_df = test_df.copy()
            test_section_df[section] = test_section_df[section].astype(str)
            
            for target in [
                "days_worked_past_week",
                ]:
                
                task = target
                if structured_features:
                    task += ":" + "".join(s[0].upper() for s in structured_features)
                
                if liwc_categories:
                    task += ":LIWC" 

                if readability_features:
                    task += ":Read"

                if chief_complaints:
                    task += ":CC"
                
                if balanced:
                    task += ":balanced"

                # Load the best model and vectorizer
                best_model = pickle.load(
                    open(
                        f"{output_path}/{model_type}/{task}_{feature_used}_{section}_{model_type}_predictor.pkl",
                        "rb",
                    )
                )

                test_features = []
                if feature_used in ["both", "notes"]:
                    vectorizer = pickle.load(
                        open(
                            f"{output_path}/{model_type}/{task}_{feature_used}_{section}_{model_type}_vectorizer.pkl",
                            "rb",
                        )
                    )
                    test_note_X = vectorizer.transform(test_section_df[section])
                    test_features.append(test_note_X)

                if feature_used in ["both", "structured"]:
                    scaler = pickle.load(
                        open(
                            f"{output_path}/{model_type}/{task}_{feature_used}_{section}_{model_type}_scaler.pkl",
                            "rb",
                        )
                    )
                    imputer = pickle.load(
                        open(
                            f"{output_path}/{model_type}/{task}_{feature_used}_{section}_{model_type}_imputer.pkl",
                            "rb",
                        )
                    )
                    cols = structured_features
                    if liwc_categories:
                        cols = cols + liwc_categories
                    if readability_features:
                        cols = cols + readability_features
                    test_standardized_X = imputer.transform(
                        test_section_df[cols]
                    )
                    test_standardized_X = scaler.transform(test_standardized_X)

                    test_standardized_X = csr_matrix(test_standardized_X)
                    print("get standardized features")
                    print(
                        "test standardized shape",
                        test_standardized_X.shape,
                        "type",
                        type(test_standardized_X),
                    )

                    test_features.append(csr_matrix(test_standardized_X))
                if chief_complaints:
                    test_features.append(csr_matrix(test_section_df[chief_complaints].values))
                
                test_input_X = hstack(test_features)
                
                try:
                    
                    preds = best_model.predict_proba(test_input_X)[:,1]
                    print("predicting probabilities")
                except:
                    print("predicting numbers")
                    preds = best_model.predict(test_input_X)
                test_df[f"tiredness_score"] = preds
                test_df[["ed_enc_id", f"tiredness_score"]].to_csv(
                    f"{output_path}/test_tiredness_score_{feature_used}_{task}_{section}_{model_type}.csv",
                    index=False,
                )
                print(
                    "Finished predicting tiredness score. Save to:",
                    f"{output_path}/test_tiredness_score_{feature_used}_{task}_{section}_{model_type}.csv",
                )

if __name__ == "__main__":

    use_liwc = True
    use_readability = True

    balanced = True
    do_train = True
    do_predict = True


    train_df = pd.read_csv("./data/train.csv")
    test_df = pd.read_csv("./data/test.csv")

    print("Load data")
    result = pyreadr.read_r('./data/cc_full_cohort.rds')[None]
    cc = [s for s in result.columns if s.startswith("cc_")]
    train_df = train_df.merge(result, on="ed_enc_id")
    test_df = test_df.merge(result, on="ed_enc_id")
    
    md_workload = pd.read_csv("./data/workload.csv")
    train_df = train_df.merge(md_workload, on="ed_enc_id")
    test_df = test_df.merge(md_workload, on="ed_enc_id")

    feature_list = [
        "note_len", "ppl_log",
        "stopword_frac", "medicalword_frac"
    ]
    if use_liwc:
        liwc_categories = [
        'pronoun_frac', 'i_frac', 'we_frac','you_frac','shehe_frac','they_frac','ipron_frac',
        'affect_frac','posemo_frac','negemo_frac','anx_frac','anger_frac','sad_frac',
        'cogmech_frac','insight_frac','cause_frac','discrep_frac','tentat_frac','certain_frac','inhib_frac','incl_frac','excl_frac']
    else:
        liwc_categories = None

    if use_readability:
        readability_features = [
            "readability_flesch_kincaid", 
            ]
    else:
        readability_features = None

    section_list = [
            "content_sections",
        ]
    
    for model_type in ["lr_cla"]:
        if not os.path.exists(f"{output_path}/{model_type}"):
            os.mkdir(f"{output_path}/{model_type}")
        for feature_used in [ "structured", "CConly"]:
            if feature_used in [ "structured", "both"]:
                for i in range(len(feature_list), len(feature_list)+1):
                    combos = itertools.combinations(feature_list, i)
                    for structured_features in combos:
                        run(
                            train_df,
                            test_df,
                            section_list,
                            do_train,
                            do_predict,
                            model_type,
                            feature_used,
                            list(structured_features),
                            cc, #use cc features
                            liwc_categories,
                            readability_features,
                            balanced,
                        )
            else:
                # run_boostrap(
                run(
                    train_df,
                    test_df,
                    section_list,
                    do_train,
                    do_predict,
                    model_type,
                    feature_used,
                    None, #structured_features
                    cc, #use cc features
                    None, #liwc_categories,
                    None, #readability_features,
                    balanced,
                )