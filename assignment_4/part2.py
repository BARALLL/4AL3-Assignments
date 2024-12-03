import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader

seed = 21
torch.manual_seed(seed)
np.random.seed(seed)

initial_compas_df = pd.read_csv("compas-scores.csv")
initial_compas_df = initial_compas_df.sample(frac=1, random_state=seed).reset_index(drop=True)

# remove columns that have more than x% of NaN values
NAN_PERCENTAGE_THRESHOLD = 0.1
threshold = len(initial_compas_df) * NAN_PERCENTAGE_THRESHOLD
cols_to_drop = initial_compas_df.columns[initial_compas_df.isna().sum() > threshold]
cleaned_compas_df = initial_compas_df.drop(columns=cols_to_drop)
cleaned_compas_df = cleaned_compas_df.dropna(how="any")

processed_df = pd.DataFrame()

# cleaned compas date time df
# ccdt_df = cleaned_compas_df.apply(lambda x: pd.to_datetime(x, errors='ignore'))
#? why depreciating errors='ignore'? now we have to do this...
tmp_df = cleaned_compas_df.copy()
for col in cleaned_compas_df.select_dtypes(include="O"):
    try:
        timestamps = pd.to_datetime(tmp_df[col])
        # processed_df[f'{col}_year'] = timestamps.dt.year
        processed_df[f'{col}_month'] = timestamps.dt.month
        processed_df[f'{col}_day'] = timestamps.dt.day
        # processed_df[f'{col}_hour'] = timestamps.dt.hour
        tmp_df.drop(col, inplace=True, axis=1)
    except:
        pass

# as names have no inherent logical order, we would need to apply one hot encoding
# they are almost as many unique names than samples in our dataset,
# meaning that this feature wont help our model learn. As such, we should drop it
# using the same logic, c_case_number is unique for every sample, so we should drop it as well

# v_type_of_assessment and type_of_assessment contains only one value,
# respectively 'Risk of Violence' and 'Risk of Recidivism',
# so we can drop it for now

tmp_df.drop(['name', 'first', 'last', 'c_case_number', 'v_type_of_assessment', 'type_of_assessment'], inplace=True, axis=1)

hm = {"Low": 0, "Medium": 0, "High": 1}
processed_df["score"] = tmp_df["score_text"].map(hm.get)
tmp_df.drop(['score_text', 'v_score_text'], inplace=True, axis=1)

# the dataset contains only Male and Female,
# with an extreme unbalance: 9336 Male & 2421 Female

# remaining features
def encode_ordinal(tmp_df, processed_df):
    tmp_df["score"] = processed_df["score"]
    for feature in tmp_df.select_dtypes(include="O").columns:
        grouped_data = tmp_df[[feature, "score"]].groupby(feature).mean()
        sorted_data = grouped_data.sort_values(by="score", ascending=False)
        sorted_data["rank"] = sorted_data["score"].rank(method="min", ascending=True)
        rank_dict = sorted_data["rank"].to_dict()

        processed_df[feature] = tmp_df[feature].map(rank_dict.get)

    tmp_df.drop(tmp_df.select_dtypes(include="O").columns, inplace=True, axis=1)
    tmp_df.drop("score", inplace=True, axis=1)

encode_ordinal(tmp_df, processed_df)

def encode_onehot(tmp_df, processed_df):
    processed_df_copy = processed_df.copy()
    nominal_features = tmp_df.select_dtypes(include="O").columns
    for feature in nominal_features:
        dummies = pd.get_dummies(tmp_df[feature], prefix=feature)
        processed_df_copy = pd.concat([processed_df_copy, dummies], axis=1)

    tmp_df.drop(nominal_features, axis=1, inplace=True)
    return processed_df_copy

# processed_df = encode_onehot(tmp_df, processed_df)


from sklearn.preprocessing import MinMaxScaler

# remove numeric feature that does not add value
tmp_df.drop(['id'], inplace=True, axis=1)

scaler = MinMaxScaler().set_output(transform="pandas")
processed_df[tmp_df.columns] = tmp_df
processed_df = scaler.fit_transform(processed_df)

# print(processed_df.head(5))

from sklearn.model_selection import train_test_split

processed_df = processed_df.astype('float32')

# p = processed_df.drop(['decile_score', 'decile_score.1', 'v_decile_score'], axis=1)

d = ['year', 'month', 'day', 'hour']
c = [name for name in processed_df.columns if any(m in name for m in d)]
p = processed_df.drop(c, axis=1)

f = plt.figure(figsize=(10, 10))
plt.matshow(p.corr(), fignum=f.number)
plt.xticks(range(p.select_dtypes(['number']).shape[1]), p.select_dtypes(['number']).columns, fontsize=12, rotation=90)
plt.yticks(range(p.select_dtypes(['number']).shape[1]), p.select_dtypes(['number']).columns, fontsize=12)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
# plt.show()

# visualizing the correlation matrix shows that each of those couple are higly correlated:
# - age with age_cat (expected)
# - is_recid with r_charge_degree (expected because there could only be a charge if there's a recid.
#   r_charge_degree carries both information, so we will keep this one)
# - decile_score.1 with decile_score and with v_decile_score (we will keep decile_score as it is more correlated to score than the other 2)
# in order to prevent the present overfitting, we will drop one of each couple

p = processed_df.drop(['age_cat', 'is_recid', 'decile_score.1', 'v_decile_score'], axis=1)

# X_train, X_test, y_train, y_test = train_test_split(
#     p.drop('score', axis=1), p['score'], test_size=0.2, stratify=processed_df["score"], random_state=seed, shuffle=True
# )

# equivalent to a classical train_test_split but allow keeping track of information prior to encoding 
def cst_split(p):
    indices = np.arange(len(p))

    train_indices, test_indices = train_test_split(
        indices, 
        test_size=0.2, 
        stratify=p["score"], 
    )

    X_train = p.drop('score', axis=1).iloc[train_indices]
    y_train = p['score'].iloc[train_indices]
    X_test = p.drop('score', axis=1).iloc[test_indices]
    y_test = p['score'].iloc[test_indices]

    X_train = torch.tensor(X_train.values)
    X_test = torch.tensor(X_test.values)
    y_train = torch.tensor(y_train.values)
    y_test = torch.tensor(y_test.values)

    return test_indices,X_train,X_test,y_train,y_test

test_indices, X_train, X_test, y_train, y_test = cst_split(p)
input_dim = X_train.shape[1]
output_dim = 2

class BinaryLogisticRegression(nn.Module):
    activation_function = nn.ReLU

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train(epochs, net, X_train, y_train, loss, optimizer):
    for _ in range(epochs):
        net.train()
        epoch_training_loss = 0
        shuffled_idxes = np.random.permutation(len(X_train))
        for input, label in zip(X_train[shuffled_idxes], y_train[shuffled_idxes]):
            optimizer.zero_grad()
            output = net(input)
            training_loss = loss(output, label.long())
            training_loss.backward()
            optimizer.step()
            epoch_training_loss += training_loss.item()
        compute_accuracy(X_train, y_train, net, "Train")
        print("Training Loss:", epoch_training_loss / (1 or len(X_train)))

def compute_accuracy(X_train, y_train, model, dataset_name=""):
    num_correct = 0
    
    model.eval()
    with torch.no_grad():
        for samples, y_true in zip(X_train, y_train):
            _, y_pred = torch.max(model(samples), -1)
            num_correct += (y_true == y_pred).sum().item()
    accuracy = 100 * num_correct / (len(X_train) or 1)
    print(f'{dataset_name} Accuracy = {accuracy:.2f}%')

def compute_eq_odds(y_true, y_pred, sensitive_attribute):
    attribs = sensitive_attribute.unique()
    rates = []
    for attrib in attribs:
        rates.append(compute_rates(y_true, y_pred, sensitive_attribute, attrib))

    # Calculate the average TPR and FPR across all groups
    avg_tpr = np.mean([rate['tpr'] for rate in rates])
    avg_fpr = np.mean([rate['fpr'] for rate in rates])

    # Calculate the difference in TPR and FPR for each group compared to the average
    eq_odds_diffs = {}
    for rate in rates:
        eq_odds_diffs[rate['val']] = {
            'tpr_diff': abs(rate['tpr'] - avg_tpr),
            'fpr_diff': abs(rate['fpr'] - avg_fpr)
        }

    # Calculate the maximum difference in TPR and FPR across all groups
    max_tpr_diff = max(eq_odds_diffs.values(), key=lambda x: x['tpr_diff'])['tpr_diff']
    max_fpr_diff = max(eq_odds_diffs.values(), key=lambda x: x['fpr_diff'])['fpr_diff']

    print("Maximum difference in TPR:", max_tpr_diff)
    print("Maximum difference in FPR:", max_fpr_diff)

    # Plot the differences in TPR and FPR for each group
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(list(x if x is not None else "" for x in eq_odds_diffs.keys()), [rate['tpr_diff'] for rate in eq_odds_diffs.values()])
    plt.xlabel("Sensitive Attribute")
    plt.ylabel("Difference in TPR")
    plt.title("Difference in TPR")
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.bar(list(x if x is not None else "" for x in eq_odds_diffs.keys()), [rate['fpr_diff'] for rate in eq_odds_diffs.values()])
    plt.xlabel("Sensitive Attribute")
    plt.ylabel("Difference in FPR")
    plt.title("Difference in FPR")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    return eq_odds_diffs

def compute_rates(y_true, y_pred, sensitive_attribute, val):
    tp = sum((sensitive_attribute == val) & (y_pred == 1) & (y_true == 1))
    fn = sum((sensitive_attribute == val) & (y_pred == 0) & (y_true == 1))
    fp = sum((sensitive_attribute == val) & (y_pred == 1) & (y_true == 0))
    tn = sum((sensitive_attribute == val) & (y_pred == 0) & (y_true == 0))

    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0

    return {'val': val, 'tpr': tpr, 'fpr': fpr}


def get_decoder_race(cleaned_compas_df, p, test_indices):
    groups = cleaned_compas_df['race'].unique()
    mapping = {} # map back from ordinal encoding to original group
    mapped_races = set()

    for sample_id in test_indices:
        if sample_id not in cleaned_compas_df.index: # should never be the case
            continue
    
        race = cleaned_compas_df.at[sample_id, 'race']
    
        if race not in mapped_races:
            mapping[p.loc[sample_id, 'race']] = race
            mapped_races.add(race)
        
            if len(mapped_races) == len(groups):
                break
    else:
        print("Not all groups are present in the test set")

    return mapping

def subsample(p):
    arr = np.unique(cleaned_compas_df['race'], return_counts=True)
    freqs = {k: v for k,v in zip(*arr)}
    smallest_group = min(freqs, key=freqs.get)
    subset_smallest_group = p.loc[cleaned_compas_df['race'] == smallest_group]

    max_cnt = len(subset_smallest_group)
    samples_from_groups = [subset_smallest_group]
    for group in freqs.keys():
        if group == smallest_group: continue

        filtered_df = p.loc[cleaned_compas_df['race'] == group]

        if len(filtered_df) > max_cnt:
            test_size = (len(filtered_df) - max_cnt) / len(filtered_df)
        else:
            test_size = 0.0

        _, selection = train_test_split(filtered_df, test_size=test_size, stratify=filtered_df['score'], random_state=seed)

        # ensure the selection has the maximum count of samples
        selection = selection.head(max_cnt)

        samples_from_groups.append(selection)
    
    subset = pd.concat(samples_from_groups)

    return subset

from sklearn.linear_model import LogisticRegression
def run(X_train, y_train, X_test, y_test, learning_rate, epochs, sensitive_attribute):
    net = BinaryLogisticRegression().to(torch.device('cpu'))
    loss = nn.CrossEntropyLoss()

    # l2 reg (weight_decay) to reduce overfitting
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate) # weight_decay=1e-2

    train(epochs, net, X_train, y_train, loss, optimizer)
    compute_accuracy(X_test, y_test, net, "Test")

    y_pred = np.array([torch.max(net(sample.unsqueeze(0)), dim=1)[1].item() for sample in X_test])

    compute_eq_odds(y_pred, y_test.cpu().numpy(), sensitive_attribute)



learning_rate = 1e-2
epochs = 3 #! risk of overfitting if higher

test_indices, X_train, X_test, y_train, y_test = cst_split(p)
mapping = get_decoder_race(cleaned_compas_df, p, test_indices)
sensitive_attribute = p['race'].iloc[test_indices].map(mapping.get)
run(X_train, y_train, X_test, y_test, learning_rate, epochs, sensitive_attribute)


subset = subsample(p)
test_indices, X_train, X_test, y_train, y_test = cst_split(subset)
mapping = get_decoder_race(cleaned_compas_df, p, test_indices)
sensitive_attribute = p['race'].iloc[test_indices].map(mapping.get)
run(X_train, y_train, X_test, y_test, learning_rate, epochs, sensitive_attribute)