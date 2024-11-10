# Author: Swati Mishra
# Created: Sep 23, 2024
# License: MIT License
# Purpose: This python file includes boilerplate code for Assignment 3

# Usage: python support_vector_machine.py

# Dependencies: None
# Python Version: 3.6+

# Modification History:
# - Version 1 - added boilerplate code
# - Version 2 - extensive modification for the purpose of the assignment

# References:
# - https://www.python.org/dev/peps/pep-0008/
# - Python Documentation

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle

np.random.seed(0)

class svm_:
    def __init__(
        self,
        learning_rate,
        epoch,
        C_value,
        X,
        Y,
        X_val,
        y_val,
        early_stopping_threshold=1e-2,
        early_stopping_patience=1,
        regularization_param=0,
        early_stopping_comparison_window=1,
    ):
        # initialize the variables
        self.input = X
        self.target = Y
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.C = C_value
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.reg_param = regularization_param  # * 0.01 is a good value
        self.random_state = 0
        self.early_stopping_comparison_window = early_stopping_comparison_window
        self.X_val = X_val
        self.y_val = y_val

        # initialize the weight matrix based on number of features
        # bias and weights are merged together as one matrix

        # self.weights = np.zeros(X.shape[1])
        self.weights = np.random.default_rng(seed=self.random_state).uniform(-0.1, 0.1, X.shape[1])
        # self.weights = np.random.randn(X.shape[1]) * 0.01

    def pre_process(self):
        # using StandardScaler to normalize the input
        scalar = StandardScaler().fit(self.input)
        X_ = scalar.transform(self.input)

        Y_ = self.target
        return X_, Y_

    # the function return gradient for 1 instance -
    # stochastic gradient decent
    def compute_gradient(self, X, Y):
        # hinge loss
        Y = Y.flatten()
        hinge_distances = 1 - (Y * np.dot(X, self.weights))
        # vectorized version of for loop if distance > 0 total_distance += -self.C * Y[i] * X[i]
        total_distance = -self.C * np.dot(X.T, Y * (np.sign(hinge_distances)))
        total_distance = total_distance / X.shape[0] + (0.5*self.reg_param) * self.weights
        return total_distance

    # def compute_gradients(self, X, Y):

    def compute_loss(self, X, Y, reg=True):
        # calculate hinge loss
        # hinge loss implementation- start
        #! .ravel() changes everything...
        hinge_distances = 1 - (Y.ravel() * np.dot(X, self.weights).ravel())
        hinge_loss = np.maximum(0, hinge_distances)
        regularization = (0.5*self.reg_param) * np.dot(self.weights, self.weights)
        loss = np.mean(hinge_loss)

        # Part 1
        # hinge loss implementatin - end

        return loss + (regularization if reg else 0)

    def stochastic_gradient_descent(self, X, Y):

        self.mini_batch_gradient_descent(X, Y, batch_size=1, early_stopping=True)
        return

        samples = 0

        train_losses = []
        val_losses = []

        early_stopping_patience_counter = 0

        best_epoch = 0
        stopped = False

        # execute the stochastic gradient descent function for defined epochs
        for epoch in range(self.epoch):

            # shuffle to prevent repeating update cycles
            features, output = shuffle(X, Y, random_state=self.random_state)
            val_features, val_output = shuffle(
                self.X_val, self.y_val, random_state=self.random_state
            )

            for i, feature in enumerate(features):
                gradient = self.compute_gradient(
                    np.array([feature]), np.array([output[i]])
                )
                self.weights = self.weights - (self.learning_rate * gradient)

            # compute train and val loss
            train_loss = self.compute_loss(features, output)
            val_loss = self.compute_loss(val_features, val_output)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            samples+=features.shape[0]

            # print epoch if it is equal to thousand - to minimize number of prints
            if epoch % (self.epoch // 10) == 0:
                print(
                    f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}"
                )
                self.compute_metrics(self.X_val, self.y_val)

            # check for convergence -start

            # Part 1
            if (
                epoch > 0
                and len(val_losses) >= self.early_stopping_comparison_window == 0
            ):
                if (
                    abs(
                        val_losses[-1]
                        - val_losses[-self.early_stopping_comparison_window]
                    )
                    < self.early_stopping_threshold
                ):
                    early_stopping_patience_counter += 1
                else:
                    early_stopping_patience_counter = 0

                if (
                    not stopped
                    and early_stopping_patience_counter >= self.early_stopping_patience
                ):
                    print(f"Early stopping at epoch {epoch}")
                    best_epoch = epoch
                    stopped = True
                    # break

        print("The minimum number of iterations taken are:", best_epoch)

        # check for convergence - end

        # below code will be required for Part 3

        # Part 3

        print("Training ended...")
        print("weights are: {}".format(self.weights))

        # below code will be required for Part 3
        print("The minimum number of samples used are:", samples)

    def mini_batch_gradient_descent(self, X, Y, batch_size, early_stopping=False):
        # mini batch gradient decent implementation - start
        # Part 2
        iterations = 0

        train_losses = []
        val_losses = []

        early_stopping_patience_counter = 0

        best_epoch = 0
        best_weights = None
        min_iterations = 0
        stopped = False

        it_counter = 1

        # execute the gradient descent function for defined epochs
        for epoch in range(self.epoch):
            if stopped: break
            # shuffle to prevent repeating update cycles
            features, output = shuffle(X, Y, random_state=self.random_state)

            # iterations
            for i in range(0, features.shape[0], batch_size):
                X_batch = features[i : i + batch_size]
                y_batch = output[i : i + batch_size]

                gradient = self.compute_gradient(X_batch, y_batch)

                self.weights = self.weights - (self.learning_rate * gradient)
                iterations += 1
                if not early_stopping: min_iterations = iterations

                # compute train and val loss
                train_loss = self.compute_loss(features, output, reg=False)
                val_loss = self.compute_loss(self.X_val, self.y_val, reg=False)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # print every 1/10th of epoch size
                if i >= (features.shape[0] // 10)*it_counter:
                    it_counter+=1
                    print(
                        f"Iteration {iterations}:{f" Minibatches {min(i, features.shape[0])}," if batch_size>1 else ""} Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}"
                    )
                    # self.compute_metrics(self.X_val, self.y_val)

                # Part 1
                # early stopping mechanism
                if (
                    early_stopping
                    and not stopped
                    and iterations > 0
                    and len(val_losses) > self.early_stopping_comparison_window
                ):
                    if (
                        val_losses[-1]
                        >= (val_losses[-1 - self.early_stopping_comparison_window]
                        - self.early_stopping_threshold)
                    ):
                        early_stopping_patience_counter += 1
                        if early_stopping_patience_counter >= self.early_stopping_patience:
                            print(f"Early stopping at iteration {iterations}")
                            best_epoch = epoch
                            min_iterations = iterations
                            best_weights = self.weights
                            stopped = True
                            # print(
                            #     f"Iteration {iterations}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}"
                            # )
                            # self.compute_metrics(self.X_val, self.y_val)
                            break
                    else:
                        early_stopping_patience_counter = 0

            it_counter = 0
            train_losses=train_losses[-early_stopping_patience_counter:]
            val_losses=val_losses[-early_stopping_patience_counter:]

        print("The minimum number of iterations taken are:", min_iterations)

        # mini batch gradient decent implementation - end

        print("Training ended...")
        # self.weights = best_weights if early_stopping else self.weights
        print("weights are: {}".format(self.weights))

    def sampling_strategy(self, X, Y):
        # implementation of sampling strategy - start

        # Part 3
        losses = [
            self.compute_loss(np.array([feature]), np.array([output]), reg=False)
            for feature, output in zip(X, Y)
        ]

        # select the sample with the smallest loss
        # argmax for largest loss
        next_sample_idx = np.argmin(losses)

        # implementation of sampling strategy - end

        return next_sample_idx

    def predict(self, X_test):
        # compute predictions on test set
        predicted_values = [
            np.sign(np.dot(X_test[i], self.weights)) for i in range(X_test.shape[0])
        ]
        return predicted_values

    def compute_metrics(self, X_test, Y_test):
        predicted_values = self.predict(X_test)
        # compute accuracy
        accuracy = accuracy_score(Y_test, predicted_values)
        print(f"Accuracy on test dataset: {accuracy:.4f}")

        # compute precision - start
        # Part 2
        precision = precision_score(Y_test, predicted_values)
        print(f"Precision on test dataset: {precision:.4f}")
        # compute precision - end

        # compute recall - start
        # Part 2
        recall = recall_score(Y_test, predicted_values)
        print(f"Recall on test dataset: {recall:.4f}")
        # compute recall - end
        return accuracy, precision, recall


def part_1(X_train, X_val, y_train, y_val):
    # model parameters - try different ones
    C = 5
    learning_rate = 2e-3
    epoch = 1
    reg_param = 1e-2

    # intantiate the support vector machine class above
    my_svm = svm_(
        learning_rate=learning_rate,
        epoch=epoch,
        C_value=C,
        X=X_train,
        Y=y_train,
        X_val=X_val,
        y_val=y_val,
        early_stopping_patience=5,
        regularization_param=reg_param
    )

    # preprocess
    X_train, y_train = my_svm.pre_process()

    # train model
    my_svm.stochastic_gradient_descent(X_train, y_train)

    return my_svm


def part_2(X_train, X_val, y_train, y_val):
    C = 1
    learning_rate = 3e-2
    epoch = 2
    reg_param = 1e-2

    # intantiate the support vector machine class above
    my_svm = svm_(
        learning_rate=learning_rate,
        epoch=epoch,
        C_value=C,
        X=X_train,
        Y=y_train,
        X_val=X_val,
        y_val=y_val,
        regularization_param=reg_param,
        early_stopping_patience=5
    )

    # preprocess
    X_train, y_train = my_svm.pre_process()

    # train model
    my_svm.mini_batch_gradient_descent(X_train, y_train, batch_size=16)

    return my_svm


def part_3(X_train, X_val, y_train, y_val):
    C = 1
    learning_rate = 1e-3
    epoch = 1
    reg_param = 1e-2
    satisfactory_acc_performance = 0.9 # could be higher but produce a lot of prints (~18 per training)

    X_train, X_unlabeled, y_train, y_unlabeled = train_test_split(
        X_train, y_train, test_size=0.8, random_state=42, shuffle=True, stratify=y_train
    )

    scalar = StandardScaler().fit(X_unlabeled)
    X_unlabeled = scalar.transform(X_unlabeled)

    accuracy = 0
    while len(y_unlabeled) > 0 and accuracy < satisfactory_acc_performance:
        my_svm = svm_(
            learning_rate=learning_rate,
            epoch=epoch,
            C_value=C,
            X=X_train,
            Y=y_train,
            X_val=X_val,
            y_val=y_val,
            regularization_param=reg_param,
            early_stopping_patience=5
        )

        # preprocess
        X_train, y_train = my_svm.pre_process()

        # train model
        my_svm.mini_batch_gradient_descent(
            X_train, y_train, batch_size=1, early_stopping=False
        )

        accuracy, _, _ = my_svm.compute_metrics(X_val, y_val)

        # select samples for training
        next_sample_idx = my_svm.sampling_strategy(X_unlabeled, y_unlabeled)

        # add newly labeled instance to training set
        X_train = np.vstack((X_train, X_unlabeled[next_sample_idx]))
        y_train = np.append(y_train, y_unlabeled[next_sample_idx])

        # remove newly labeled instance from unlabeled set
        # (can be "slow" (still fast because in C) because arrays are immutable so np.delete recreate the array)
        X_unlabeled = np.delete(X_unlabeled, next_sample_idx, axis=0)
        y_unlabeled = np.delete(y_unlabeled, next_sample_idx, axis=0)

    return my_svm


# Load datapoints in a pandas dataframe
print("Loading dataset...")
data = pd.read_csv("data.csv")

# drop first and last column
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

# segregate inputs and targets

# inputs
X = data.iloc[:, 1:]

# add column for bias
X.insert(loc=len(X.columns), column="bias", value=1)
X_features = X.to_numpy()

# converting categorical variables to integers
# - this is same as using one hot encoding from sklearn
# benign = -1, melignant = 1
category_dict = {"B": -1.0, "M": 1.0}
# transpose to column vector
Y = np.array([(data.loc[:, "diagnosis"]).to_numpy()]).T
Y_target = np.vectorize(category_dict.get)(Y)

# data imbalance, 357 benign, 212 melignant

# split data into train and test set using sklearn feature set
print("splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_features, Y_target, test_size=0.2, shuffle=True, random_state=42, stratify=Y_target
)

# split training data into train and validation set using sklearn feature set
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=True, random_state=42, stratify=y_train
)

# normalize validation set
scalar = StandardScaler().fit(X_val)
X_val = scalar.transform(X_val)

# normalize the test set separately
scalar = StandardScaler().fit(X_test)
X_Test_Norm = scalar.transform(X_test)


print('\n', '='*20, "part 1", '='*20)
my_svm1 = part_1(X_train, X_val, y_train, y_val)
print("Testing part 1 model accuracy...")
my_svm1.compute_metrics(X_Test_Norm, y_test)

print('\n', '='*20, "part 2", '='*20)
my_svm2 = part_2(X_train, X_val, y_train, y_val)
print('\n','='*50)
print("Testing part 2 model accuracy...")
my_svm2.compute_metrics(X_Test_Norm, y_test)

print('\n', '='*20, "part 3", '='*20)
my_svm3 = part_3(X_train, X_val, y_train, y_val)
print('\n','='*50)
print("Testing part 3 model accuracy...")
my_svm3.compute_metrics(X_Test_Norm, y_test)

print('\n', '='*20, "re-display model metrics", '='*20)
print("part 1")
my_svm1.compute_metrics(X_Test_Norm, y_test)

print("\n part 2")
my_svm2.compute_metrics(X_Test_Norm, y_test)

print("\n part 3")
my_svm3.compute_metrics(X_Test_Norm, y_test)
