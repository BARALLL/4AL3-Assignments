import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sklearn
from sklearn.metrics import make_scorer, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#! define SVC, GridSearchCV, and cross-validation parameters
# change shuffle to False if you want to use the data order from data_order.npy
shuffle = False
random_state = None if not shuffle else 0
np.random.seed(random_state)

model_kwargs = {
    "C": 6,
    "tol": 1e-8,
    # 'gamma': 0.075, # value used in the paper but is no longer applicable according to the experiments
    # 'class_weight': {0: 1, 1: 2},
    "class_weight": "balanced",
    "kernel": "rbf",
}

# define the parameter grid for GridSearchCV
param_grid = {
    "C": [i / 10 for i in range(1, 100)],
    "kernel": [
        "rbf"
    ],  # test performed "by hand", 'rbf' is by far the most efficient, and is also the one used in the research paper
    "gamma": ["scale", "auto", 0.1, 0.075, 0.01, 0.001],
    "class_weight": ["balanced", None],
    "tol": [1e-8],
}


# the (X or 1) is in the case where X is 0, we dont want to divide by 0...
def TSS(tp: int, fp: int, fn: int, tn: int):
    return tp / ((tp + fn) or 1) - fp / ((fp + tn) or 1)


def tss_score(y, y_pred):
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    return TSS(tp, fp, fn, tn)


n_splits = 5
cv_kwargs = {
    "scoring": make_scorer(tss_score),
    # 'cv': 5
    "cv": StratifiedKFold(
        n_splits=n_splits, random_state=random_state, shuffle=shuffle
    ),
}


# from the itertools package that is part of python
# https://docs.python.org/3/library/itertools.html#itertools.combinations
# since additional librairies arent allowed, i dont want to take any risk importing it directly
# as this could have been written from scratch (not a very difficult algorithm)
# i could have listed the combinations by hand like below, but the would not have been very clean
# (0,), (1,), (2,), (3,), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
# (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 2, 3)
def combinations(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


def load_data_from_path(dataset_path):
    # * neg and pos are composed of HARP_id, timestamp, magnitude level
    neg_data = np.load(os.path.join(dataset_path, "neg_class.npy"), allow_pickle=True)
    pos_data = np.load(os.path.join(dataset_path, "pos_class.npy"), allow_pickle=True)

    # * neg_features_main_timechange is composed of 18 features - Main feature set (FS -I)
    # * then between 19-90, they are 4 timestamp change of the 18 previous features - Time Change Feature (FS-II)
    # taking 1st feature, USFLUX, its changes are located in range 19-23 which correspond respectively to:
    # - 48 hours prior to 24 hours prior to the peak time
    # - 42 hours prior to 24 hours prior
    # - 36 hours prior to 24 hours prior
    # - 30 hours prior to 24 hours prior
    neg_features_main_timechange = np.load(
        os.path.join(dataset_path, "neg_features_main_timechange.npy"),
        allow_pickle=True,
    )
    pos_features_main_timechange = np.load(
        os.path.join(dataset_path, "pos_features_main_timechange.npy"),
        allow_pickle=True,
    )
    neg_features_main = neg_features_main_timechange[:, :19]
    pos_features_main = pos_features_main_timechange[:, :19]
    neg_features_timechange = neg_features_main_timechange[:, 19:]
    pos_features_timechange = pos_features_main_timechange[:, 19:]

    # * Historical Activity Feature (FS-III)
    # activity history: sum of scores of past activity from 24 to 48 hours prior
    neg_features_historical = np.load(
        os.path.join(dataset_path, "neg_features_historical.npy"), allow_pickle=True
    )
    pos_features_historical = np.load(
        os.path.join(dataset_path, "pos_features_historical.npy"), allow_pickle=True
    )

    # * Max Min Feature (FS-IV)
    # same features as Main feature set (FS -I) but min val substacted from max val
    #  from 24 hours to 48 hours prior
    neg_features_maxmin = np.load(
        os.path.join(dataset_path, "neg_features_maxmin.npy"), allow_pickle=True
    )
    pos_features_maxmin = np.load(
        os.path.join(dataset_path, "pos_features_maxmin.npy"), allow_pickle=True
    )

    return (
        neg_features_main,
        pos_features_main,
        neg_features_timechange,
        pos_features_timechange,
        neg_features_historical,
        pos_features_historical,
        neg_features_maxmin,
        pos_features_maxmin,
    )


# generate FS-X style label from combination for each combinations
def get_label_from_comb(comb):
    number_to_roman = ["I", "II", "III", "IV"]
    return "".join("FS-" + number_to_roman[i] + ", " for i in comb)[:-2]


def input_cleaning(data_ndarrays):
    # vertically stack negative and positive vectors
    for i, fs in enumerate(data_ndarrays):
        data_ndarrays[i] = np.vstack(list(fs))

    # remove missing values
    missing_values = np.array([False] * data_ndarrays[-1].shape[0])
    for i, fs in enumerate(data_ndarrays):
        missing_values = missing_values & np.any(~np.isfinite(data_ndarrays[i]), axis=1)

    # remove rows containing missing values
    for i, fs in enumerate(data_ndarrays):
        data_ndarrays[i] = data_ndarrays[i][~missing_values]

    # transform data_ndarrays to tuple to make sure no further modification are performed
    data_ndarrays = tuple(data_ndarrays)

    return data_ndarrays


def scale_input(data_ndarrays):
    # min max scaler as data does not follow a gaussian distribution (worked best as of the experiments)
    return list(MinMaxScaler().fit_transform(fs) for fs in data_ndarrays)


# create input data given combinaison of feature set
def input_data_from_comb(data_ndarrays, data_idxes):
    X = np.hstack(list(data_ndarrays[idx] for idx in data_idxes))
    return X[data_order] if (not shuffle and X.shape[0] >= data_order.shape[0]) else X


# generator that yield every data combination for easy iteration
def generate_comb_datas(X):
    idxes_to_choose_from = range(len(X))
    for i in range(1, 5):
        for data_idxes in combinations(idxes_to_choose_from, i):
            yield (input_data_from_comb(X, data_idxes), data_idxes)


# get mean, std and scores of cross validation for each fs combinations
def get_fs_scores(random_state, data_ndarrays, y):
    fs_scores = {}
    for input_comb, data_idxes in generate_comb_datas(data_ndarrays):
        clf = SVC(random_state=random_state, **model_kwargs)
        scores = cross_val_score(clf, input_comb, y, **cv_kwargs)
        fs_scores[data_idxes] = (scores.mean(), scores.std(), scores)
    return fs_scores


# return a trained model for each fs combinations
def get_fs_models(random_state, data_ndarrays, y):
    fs_models = {}
    for input_comb, data_idxes in generate_comb_datas(data_ndarrays):
        X_train, X_test, y_train, y_test = train_test_split(
            input_comb, y, random_state=random_state, shuffle=shuffle
        )
        clf = SVC(random_state=random_state, **model_kwargs)
        clf.fit(X_train, y_train)
        fs_models[data_idxes] = clf
    return fs_models


# average k-fold scores for each feature set combination by running cross-validation multiple times and taking the mean of the scores for each fold
def fs_avg(data_ndarrays, y, AVG_ON=10):
    tmp = {idxes: [] for (_, idxes) in generate_comb_datas(data_ndarrays)}
    for i in range(AVG_ON):
        curr_scores = get_fs_scores(i, data_ndarrays, y)
        for comb in curr_scores.keys():
            tmp[comb].extend(curr_scores[comb][2])

    return {
        k: (np.mean(v), np.std(v), v, np.array(v).reshape(len(v) // 5, 5).mean(axis=0))
        for (k, v) in tmp.items()
    }


# performs a grid search to find the best hyperparameters for the given parameter grid and input data using cross-validation
def perform_gridsearch(param_grid, best_input_data, y):
    # define Grid Search
    grid_search = GridSearchCV(
        SVC(random_state=random_state),
        param_grid=param_grid,
        verbose=1,
        n_jobs=-1,
        **cv_kwargs,
    )

    # fit GridSearchCV and prints results
    grid_search.fit(best_input_data, y)
    print("Best parameters found by GridSearchCV:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    return grid_search


# visualizes the average k-fold accuracy scores for each feature set combination over multiple folds
def visualize_acc_over_folds(n_splits, avg_fs_scores, best_fs, bm_cv_scores):
    comb_to_label = {comb: get_label_from_comb(comb) for comb in avg_fs_scores.keys()}
    max_width = len(
        max(comb_to_label.values(), key=len)
    )  # get the longest label (string) to align them
    plt.figure(figsize=(10, 6))
    for comb, scores in avg_fs_scores.items():
        plt.plot(
            range(1, n_splits + 1),
            scores[3],
            linestyle="--",
            marker="o",
            label=f"{comb_to_label[comb]:<{max_width}} mean: {round(scores[0], 3)}",
        )

    # add best model
    plt.plot(
        range(1, n_splits + 1),
        bm_cv_scores,
        marker="o",
        label=f"{("best model " + get_label_from_comb(best_fs)):<{max_width}} mean: {round(bm_cv_scores.mean(), 3)}",
    )

    plt.title("SVM K-Fold Cross-Validation Accuracy averaged over multiple seeds")
    plt.xlabel("Fold Number")
    plt.ylabel("TSS")
    plt.ylim(-1, 1)
    plt.xticks(range(1, n_splits + 1))

    plt.legend(loc="lower left")

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()


# visualizes the confusion matrices for each feature set combination using the trained models from cross-validation
def visualize_cms(data_ndarrays, y, fs_models):
    comb_to_label = {comb: get_label_from_comb(comb) for comb in fs_models.keys()}

    n_feature_sets = len(fs_models)
    n_cols = 5
    n_rows = (n_feature_sets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(2 * n_cols + (n_cols - 1), 2 * n_rows + (n_rows - 1)),
    )
    axes = axes.flatten()

    for ax, (data_idxes, model) in zip(axes, fs_models.items()):
        X = input_data_from_comb(data_ndarrays, data_idxes)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=random_state, shuffle=shuffle
        )
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

        disp.plot(ax=ax)
        ax.title.set_text(f"{comb_to_label[data_idxes]}")

    for i in range(n_feature_sets, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Confusion Matrix of each feature set combinations")
    plt.tight_layout()
    plt.show()


def dataset_analysis(dataset_path):
    dataset_name = os.path.basename(dataset_path)
    print(f"performing analysis for the {dataset_name} dataset")
    # load dataset
    (
        neg_main,
        pos_main,
        neg_timechange,
        pos_timechange,
        neg_historical,
        pos_historical,
        neg_maxmin,
        pos_maxmin,
    ) = load_data_from_path(dataset_path)

    # define our feature sets
    data_ndarrays = [
        (neg_main, pos_main),
        (neg_timechange, pos_timechange),
        (neg_historical, pos_historical),
        (neg_maxmin, pos_maxmin),
    ]

    # create a column by stacking vertically 0s for all negative vectors and 1s for all positive vectors
    y = np.vstack(
        (
            np.zeros((neg_main.shape[0], 1), dtype=neg_main.dtype),
            np.ones((pos_main.shape[0], 1), dtype=pos_main.dtype),
        ),
    ).ravel()
    if not shuffle and y.shape[0] >= data_order.shape[0]:
        y = y[data_order]

    # clean and scale the input
    data_ndarrays = input_cleaning(data_ndarrays)
    data_ndarrays_norm = scale_input(data_ndarrays)

    # get k-fold scores for each feature set combination
    avg_fs_scores = fs_avg(data_ndarrays_norm, y)

    print(
        "Average K-fold mean for each feature set combination (across multiple seeds):"
    )
    for k, v in avg_fs_scores.items():
        print(
            f"{get_label_from_comb(k)}: mean {float(round(v[0],3))}, std {float(round(v[1],3))}"
        )

    # select the best performing feature sets
    best_fs = max(avg_fs_scores, key=lambda k: avg_fs_scores.get(k, ("-inf", None))[0])
    print(f"The best feature set is: {get_label_from_comb(best_fs)}")

    # now that we find the best combination of feature sets, lets find the best hyperparameters using grid search

    best_input_data = input_data_from_comb(data_ndarrays_norm, best_fs)
    grid_search = perform_gridsearch(param_grid, best_input_data, y)

    # define model with the same parameters and same feature sets that led to the best model for this dataset
    best_svc_model = SVC(**grid_search.best_params_, random_state=random_state)
    bm_cv_scores = cross_val_score(best_svc_model, best_input_data, y, **cv_kwargs)
    print(
        f"Cross-validation scores for best model using the {dataset_name} dataset:",
        bm_cv_scores,
    )
    print(
        f"Mean CV Score for best model using the {dataset_name} dataset:",
        round(bm_cv_scores.mean(), 4),
    )

    # visualize class distribution
    plt.figure(figsize=(10, 6))
    seaborn.barplot(x=[0, 1], y=[neg_main.shape[0], pos_main.shape[0]])
    plt.title(f"Class Distribution of the {dataset_name} dataset")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

    # split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        best_input_data, y, random_state=random_state, test_size=0.2, shuffle=shuffle
    )

    # fit the model for inference
    best_svc_model.fit(X_train, y_train)
    y_pred = best_svc_model.predict(X_test)

    # display confusion matrix of the best model
    fig, ax = plt.subplots(figsize=(6, 6))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["negative", "positive"]
    )
    disp.plot(ax=ax)
    plt.title(
        f"Confusion Matrix for the best model\n Feature sets used are {get_label_from_comb(best_fs)}\n Mean CV score is {round(bm_cv_scores.mean(), 3)}"
    )
    plt.show()

    # train SVC model for every combinations
    fs_models = get_fs_models(random_state, data_ndarrays_norm, y)

    visualize_acc_over_folds(n_splits, avg_fs_scores, best_fs, bm_cv_scores)
    visualize_cms(data_ndarrays_norm, y, fs_models)


local_path = os.getcwd()
dataset_2010_path = os.path.join(local_path, "data-2010-15")
dataset_2020_path = os.path.join(local_path, "data-2020-24")

data_order = np.load(os.path.join(dataset_2010_path, "data_order.npy")).ravel()

dataset_analysis(dataset_2010_path)
print("\n", "=" * 20, '\n')
dataset_analysis(dataset_2020_path)
