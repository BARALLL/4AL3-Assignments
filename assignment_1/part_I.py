import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


# normalize given vector
def vec_standardization(vec: np.ndarray):
    return (vec - vec.mean()) / vec.std()


# converts normalized regression coefficients back to original scales
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


# partial derivative of loss function (mse) with respect to betas
def compute_gradients(betas: np.ndarray, x: np.ndarray, y: np.ndarray):
    return 2 / (x.shape[0]) * ((x.T).dot(x.dot(betas) - y))


def gradient_descent(
    learning_rate: float, iterations: int, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    betas = np.random.normal(loc=0, scale=1, size=2)  # np.random.randn(2, 1)

    for _ in range(iterations):
        gradients = compute_gradients(betas, x, y)
        betas = betas - learning_rate * gradients

    return betas


def ordinary_least_squares(x, y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


# MODEL(FEATURES) = TARGET
# [function obtained after optimization] (GDP) = Hapiness

# read the data from csv and remove row if missing value in any of those 2 columns
df_data = pd.read_csv("gdp-vs-happiness.csv")[
    ["Cantril ladder score", "GDP per capita, PPP (constant 2017 international $)"]
].dropna(how="any")

# separate them into 2 vectors representing respectively the target (happiness) and the feature (GDP)
target_vec = df_data["Cantril ladder score"].to_numpy()
features_vec = df_data["GDP per capita, PPP (constant 2017 international $)"].to_numpy()

# helps GD converge faster and in case of multiple features, ensures that they all have the same impact
target_vec_norm = vec_standardization(target_vec)
features_vec_norm = vec_standardization(features_vec)

# add a column of 1s for the intercept
features_vec_norm_intercept = np.column_stack(
    (np.ones(features_vec_norm.shape[0]), features_vec_norm)
)

# 6 different sets of hyperparameters
learning_rates = [1e-1, 1e-2, 1e-2, 1e-2, 1e-4, 1e-4, 1]
iteration_counts = [10, 5, 10, 100, 10, 100000, 100]

x_space = np.linspace(0, features_vec.max())

# plot initial data
plt.plot(features_vec, target_vec, "b.")

# keep track of std and mean for both feature and target in order to un-normalize betas to plot
feature_means = [features_vec.std()]
feature_stds = [features_vec.mean()]
target_info = {"std": target_vec.std(), "mean": target_vec.mean()}

# keep track of best model
best_model = None
best_mse = float("inf")

# 1 to 1 mapping using zip for hyperparameters
for learning_rate, iteration_count in zip(learning_rates, iteration_counts):
    betas = gradient_descent(
        learning_rate, iteration_count, features_vec_norm_intercept, target_vec_norm
    )

    # eval current model
    model_eval = mse(betas, features_vec_norm_intercept, target_vec_norm)

    # un-normalize parameters
    unnormalized_betas = unnormalize_betas(
        betas, feature_means, feature_stds, target_info
    )

    # compare with mse value of current best model
    if model_eval < best_mse:
        best_mse = model_eval
        best_model = {
            "b0": unnormalized_betas[0],
            "b1": unnormalized_betas[1],
            "learning_rate": learning_rate,
            "iteration_count": iteration_count,
            "mse": model_eval,
        }

    print(
        f"b0: {unnormalized_betas[0]}, b1: {unnormalized_betas[1]}, learning rate: {learning_rate}, epochs: {iteration_count}"
    )

    # add regression line to the figure
    plt.plot(
        x_space,
        unnormalized_betas[0] + unnormalized_betas[1] * x_space,
        label=f"learning_rate: {learning_rate}, iterations: {iteration_count}",
    )

# present plot the same way as shown in the lecture
plt.xlabel("Happiness")
plt.ylabel("GDP per Capita")
plt.legend(loc="upper left")
plt.title("Cantril Ladder Score vs GDP per Capita of Countries (2018)")
# plt.savefig("relationship.png")
plt.show()

# perform OLS
betas_ols = ordinary_least_squares(features_vec_norm_intercept, target_vec_norm)

# un-normalize parameters given by OLS
unnormalized_betas_ols = unnormalize_betas(
    betas_ols, feature_means, feature_stds, target_info
)

print("")
print(
    f"betas obtained using ols: b0: {unnormalized_betas_ols[0]}, b1: {unnormalized_betas_ols[0]}, mse = {mse(betas_ols, features_vec_norm_intercept, target_vec_norm)}"
)
print(
    "best model obtained using gradient descent: ",
)
for k, v in best_model.items():
    print("\t", k, v)

print(
    "im not sure why the GD b1 is different from the b1 obtained by OLS, as the following graph shows that they overlap"
)


# plot second graph: OLS and best GD-based models on top of scatter
plt.plot(features_vec, target_vec, "b.")

plt.plot(
    x_space,
    unnormalized_betas_ols[0] + unnormalized_betas_ols[1] * x_space,
    label="OLS",
)


plt.plot(
    x_space,
    best_model["b0"] + best_model["b1"] * x_space,
    label=f"GD - learning_rate: {learning_rate}, iterations: {iteration_count}",
)

# present plot the same way as shown in the lecture
plt.xlabel("Happiness")
plt.ylabel("GDP per Capita")
plt.legend(loc="upper left")
plt.title("Cantril Ladder Score vs GDP per Capita of Countries (2018)")
# plt.savefig("relationship.png")
plt.show()


# both line overlap, that is why we only see one on the graph but by removing the GD based one we are able to see the OLS one
