import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

np.random.seed(42)


# normalize given vector
def vec_standardization(vec: np.ndarray):
    std = vec.std()
    if std == 0:
        return np.zeros_like(vec)
    return (vec - vec.mean()) / vec.std()


def unnormalize_betas(betas, feature_means, feature_stds, target_info):
    unnormalized_betas = np.zeros_like(betas)

    # unnormalize slope for each feature
    unnormalized_betas[1:] = betas[1:] * target_info["std"] / feature_stds
    # adjust intercept for each feature
    unnormalized_betas[0] = target_info["mean"] - np.sum(
        unnormalized_betas[1:] * feature_means
    )

    return unnormalized_betas


# loss function
def mse(betas: np.ndarray, x: np.ndarray, y: np.ndarray):
    return np.square(x.dot(betas) - y).mean()


def rmse(betas: np.ndarray, x: np.ndarray, y: np.ndarray):
    return np.sqrt(np.square(x.dot(betas) - y).mean())


# partial derivative of loss function (mse) with respect to betas
def compute_gradients(betas: np.ndarray, x: np.ndarray, y: np.ndarray):
    return 2 / (x.shape[0]) * ((x.T).dot(x.dot(betas) - y))


def gradient_descent(
    learning_rate: float, iterations: int, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    betas = np.random.normal(loc=0, scale=1, size=x.shape[1])  # np.random.randn(2, 1)

    for _ in range(iterations):
        gradients = compute_gradients(betas, x, y)
        betas = betas - learning_rate * gradients

    return betas


def ordinary_least_squares(x, y):
    try:
        return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    except np.linalg.LinAlgError:
        print("Some features are too highly correlated")


# read the data from csv and remove first column (vector_id)
# and each row that contains a missing value in any of the columns
df_data = pd.read_csv("training_data.csv").iloc[:, 1:].dropna(how="any")

# separate the features from the target
features_df = df_data.drop(columns="Rings")

# create simple correlation matrix
plt.figure(figsize=(len(features_df.columns), len(features_df.columns)))
corr_matrix = features_df.corr()
sn.heatmap(
    corr_matrix,
    xticklabels=corr_matrix.columns.values,
    yticklabels=corr_matrix.columns.values,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5,
)
plt.tight_layout()
# plt.savefig("correlations_betwen_features.png")
plt.show()

# from this simple correlation matrix, we can see that Diameter is extremely correlated with Length (0.99)
# as so the inverse matrix in OLS could not be computed
# all _weight variables are strongly correlated with each other (around 0.9)
# we will remove Diameter after the other visualizations
# from my experiments, the _weight variables does not seems to change RMSE too much

# visualize relationship between each couple (feature_i, output)
ncols = 3
nrows = (len(features_df.columns) // ncols) + 1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
fig.suptitle("Relationship between each feature and the target")

# loop over every feature column and plot "number of rings as function of feature_i" on a new subplot
for i, (feature_name, feature) in enumerate(df_data.drop(columns="Rings").items()):
    row_idx = i // ncols
    col_idx = i % ncols
    ax[row_idx][col_idx].scatter(feature, df_data["Rings"], s=2)
    ax[row_idx][col_idx].set_ylabel("Number of rings")
    ax[row_idx][col_idx].set_xlabel(feature_name)

    ax[row_idx][col_idx].set_title(f"Rings as a function of {feature_name}")
# extend layout to fit subplots correctly
fig.tight_layout()
# plt.savefig("relationship_with_rings.png")
plt.show()


# On this visualization, we can see that while the number of rings grows linearly with respect to some features,
# it seems to grows quadratically with respect to some others

# from the observations we can make, we can say, without jumping to conclusions, that:

# - the number of rings could grow quadratically with respect to Length, Diameter,
#       Whole_weight, Shucked_weight, Visera_weight or Shell_weight
# -  the number of rings could grow linearly with Height

# it would therefore be interesting to use a polynomial model


# visualize data distribution of each feature
ncols = 3
nrows = (len(features_df.columns) // ncols) + 1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
fig.suptitle("Histogram of each Variable")

for i, (variable_name, variable) in enumerate(df_data.items()):
    row_idx = i // ncols
    col_idx = i % ncols
    ax[row_idx][col_idx].hist(variable, bins=30)
    ax[row_idx][col_idx].set_ylabel("Frequence")
    ax[row_idx][col_idx].set_xlabel(variable_name)

    ax[row_idx][col_idx].set_title(f"Distribution of {variable_name}")
# extend layout to fit subplots correctly
fig.tight_layout()
# fig.savefig("data_distribution.png")
plt.show()

# we can see that Length and Diameter are left skewed and that
# Whole_weight, Shucked_weight, Visera_weight or Shell_weight are right skewed
# Height seems to follow a normal distribution

# using this information we can determine that IQR is a suitable method for outlier removal
# since it works well for both normal and slightly skewed distributions

# also we can see that target variable follows a really slightly left skewed distribution

# here RMSE seems the best choice as unlike MAE, which treats all errors equally,
# both RMSE and MSE emphasize minimizing bigger mistakes
# but RMSE also provides error values in the same scale as the target, making it more interpretable than MSE.

df_data = df_data.drop(
    columns=["Diameter"]
)  # 'Shucked_weight', 'Viscera_weight', 'Shell_weight'

# remove outliers using IQR method
q25 = df_data.quantile(0.25, numeric_only=False)
q75 = df_data.quantile(0.75, numeric_only=False)
iqr = q75 - q25
cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off
outliers = (df_data < lower) | (df_data > upper)
df_cleaned = df_data[~outliers.any(axis=1)]

# separate variables into 2 vectors representing respectively the target (Rings) and the features
target_vec = df_cleaned["Rings"].to_numpy()
features = df_cleaned.drop(columns="Rings").to_numpy()

# standardization technique, helps GD converge faster and in case of multiple feature,
# assure that they all have the same impact
target_norm = vec_standardization(target_vec)

polynomial_degree = 2  # keeping this to 1 is equivalent to linear regression
new_features = [features**degree for degree in range(1, polynomial_degree)]

# add polynomial features for each features
poly_features = np.concatenate(
    [features] + new_features,
    axis=1,
)
# apply standardization to features as well
poly_features_norm = poly_features.copy()
np.apply_along_axis(vec_standardization, axis=1, arr=poly_features_norm)

# add a column of 1s for the intercept
poly_features_intercept = np.column_stack(
    (np.ones(poly_features.shape[0]), poly_features)
)
poly_features_norm_intercept = np.column_stack(
    (np.ones(poly_features_norm.shape[0]), poly_features_norm)
)

# OLS or GD?

# Pros for OLS:
# OLS is computationally expensive for large datasets because matrix inversion is an O(n^3) operation
# but it is generally preferred for a dataset smaller than 10^4

# Cons for OLS:
# OLS can be inadequate in case of multicollinearity
# from the observation done on the correlation graph, we have a pretty strong presence of multicollinearity
# in general, it undermines the interpretability and add instability in coefficient estimations
# and can prevent from computing the inverse matrix of x.T
# so here OLS cannot guarantee finding the globally optimal solution

# furthermore OLS is sensitive to outliers (can undermine performance even if most of it was removed)
# also, OLS is limited to linear models and we would like to try mutlivariate polynomial regression

# so here gradient descent seems more appropriate

# keep track of std and mean for both features and target in order to un-normalize betas to plot
feature_means = np.array(
    [feature.std() for feature in poly_features_norm_intercept.T[1:]]
)
feature_stds = np.array(
    [feature.mean() for feature in poly_features_norm_intercept.T[1:]]
)
target_info = {"std": variable.std(), "mean": variable.mean()}


# i tried OLS but it did poorly and, depending on the inputs, the inverse matrix may not be computable

# while GD seems more suitable here, lets run OLS first to see how it perform
# betas_ols = ordinary_least_squares(poly_features_norm_intercept, target_norm)

# # un-normalize parameters given by OLS
# unnormalized_betas_ols = unnormalize_betas(betas_ols, feature_means, feature_stds, target_info)

# features_vec_intercept = np.column_stack(
#     (np.ones(poly_features_norm_intercept.shape[0]), poly_features)
# )

# print("Error rate using OLS method:", rmse(betas_ols, poly_features_norm_intercept, target_norm))
# print("Error rate using OLS method (unnorm):", rmse(unnormalized_betas_ols, features_vec_intercept, target_vec))


# 7 different sets of hyperparameters
learning_rates = [1e-1, 1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1e-3]
iteration_counts = [10, 5, 10, 100, 10, 1000, 10000]

# keep track of best model
best_model = None
best_mse = float("inf")

# k-fold cross validation to evaluate performance and generalization capability of our model
k = 5
indices = np.arange(len(poly_features_norm_intercept))
np.random.shuffle(indices)
folds = np.array_split(indices, k)

# 1 to 1 mapping using zip for hyperparameters
for learning_rate, iteration_count in zip(learning_rates, iteration_counts):
    evals = []
    for i in range(max(1, k)):
        # select which vector will be used for validation or training
        val_indices = folds[i]
        train_indices = np.hstack(folds[:i] + folds[i + 1 :])

        # split according to the indices
        X_train, X_val = (
            poly_features_norm_intercept[train_indices],
            poly_features_norm_intercept[val_indices],
        )
        y_train, y_val = target_norm[train_indices], target_norm[val_indices]

        betas = gradient_descent(learning_rate, iteration_count, X_train, y_train)

        unnorm_betas = unnormalize_betas(
            betas, feature_means, feature_stds, target_info
        )

        # eval current model
        model_eval = rmse(
            unnorm_betas, poly_features_intercept[val_indices], target_vec[val_indices]
        )
        evals.append(model_eval)

        # compare with mse value of current best model
        if model_eval < best_mse:
            best_mse = model_eval
            best_model = {
                "betas": unnorm_betas,
                "learning_rate": learning_rate,
                "iteration_count": iteration_count,
                "rmse": model_eval,
            }
    avg_rmse = (sum(evals)) / (k or 1)
    print(
        f"Overall performance for learning rate: {learning_rate}, epochs: {iteration_count} - RMSE {avg_rmse}"
    )

print(
    "best model using gradient descent: ",
)
for k, v in best_model.items():
    print("\t", k, v)

# the best RMSE obtained was 1.8438 with the following betas:
# betas = [  8.67399634   0.7713529  -10.68932739  -0.76439472   6.43977124
#   -0.24361187  -5.3075009    2.03669155 -16.62913198   5.28049335
#  -14.77036817  12.66896731   8.32612362]
# corresponding to
# Length,Height,Whole_weight,Shucked_weight,Viscera_weight,Shell_weight
# Length^2,Height^2,Whole_weight^2,Shucked_weight^2,Viscera_weight^2,Shell_weight^2

# most RMSE were around 2 - 2.20

# but i felt like polynomial regression was more subject to random and while
# most of the time the best model it was on par with linear regression,
# RMSE where usually higher for other model (different learning rate and epoch)


# visualize line on chart of each feature
ncols = 3
nrows = (len(features_df.columns) // ncols) + 1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
fig.suptitle("Relationship between each feature and the target")

nb_feature = len(df_data.columns)

# loop over every feature column and plot "number of rings as function of feature_i" on a new subplot
for i, (feature_name, feature) in enumerate(df_data.drop(columns="Rings").items()):
    x_space = np.linspace(0, feature.max())
    row_idx = i // ncols
    col_idx = i % ncols
    ax[row_idx][col_idx].scatter(feature, df_data["Rings"], s=2)
    ax[row_idx][col_idx].plot(
        x_space,
        best_model["betas"][0]
        + sum(
            best_model["betas"][degree + (i + 1)] * (x_space ** (degree + 1))
            for degree in range(0, polynomial_degree)
        ),
    )
    ax[row_idx][col_idx].set_ylabel("Number of rings")
    ax[row_idx][col_idx].set_xlabel(feature_name)

    ax[row_idx][col_idx].set_title(f"Rings as a function of {feature_name}")
# extend layout to fit subplots correctly
fig.tight_layout()
# plt.savefig("relationship_with_rings_betas.png")
plt.show()
