import pandas as pd
import numpy as np

def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

def train_linear_regression_reg(X, y, r=0.001):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

def divide_dataset(df, train_size=0.6, val_size=0.2, test_size=0.2, seed=42):
    total_size = train_size + val_size + test_size
    norm_train_size = train_size / total_size
    norm_val_size = val_size / total_size
    norm_test_size = test_size / total_size

    dataset_len = len(df)

    num_train_samples = int(norm_train_size * dataset_len)
    num_val_samples = int(norm_val_size * dataset_len)
    num_test_samples = int(norm_test_size * dataset_len)

    idx = np.arange(dataset_len)

    np.random.seed(seed)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:num_train_samples]]
    df_val = df.iloc[idx[num_train_samples:num_train_samples+num_val_samples]]
    df_test = df.iloc[idx[num_train_samples+num_val_samples:]]

    return df_train, df_val, df_test

csv_dataframe = pd.read_csv("AB_NYC_2019.csv")
print(csv_dataframe.head())

print(csv_dataframe["price"].value_counts())

columns = ["latitude", "longitude", "price", "minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count", "availability_365"]
subset_dataframe = csv_dataframe[columns]

for column in columns:
    print(column, subset_dataframe[column].isna().sum())

print(subset_dataframe["minimum_nights"].median())

shuffled = subset_dataframe.sample(frac=1, random_state=42)

train, val, test = divide_dataset(subset_dataframe, 0.6, 0.2, 0.2)

train_X = train[[c for c in columns if c != "price"]]
train_y = np.log1p(train["price"])
val_X = val[[c for c in columns if c != "price"]]
val_y = np.log1p(val["price"])
test_X = test[[c for c in columns if c != "price"]]
test_y = np.log1p(test["price"])

print(len(train))
print(len(val))
print(len(test))

print(train.columns)

mean_reviews_per_month = train_X["reviews_per_month"].mean()

train_X_impute_mean = train_X.copy()
train_X_impute_mean.fillna({"reviews_per_month":mean_reviews_per_month}, inplace=True)
val_X_impute_mean = val_X.copy()
val_X_impute_mean["reviews_per_month"].fillna({"reviews_per_month":mean_reviews_per_month}, inplace=True)

train_X_impute_zero = train_X.copy()
train_X_impute_zero["reviews_per_month"].fillna(0, inplace=True)
val_X_impute_zero = val_X.copy()
val_X_impute_zero["reviews_per_month"].fillna(0, inplace=True)

bias, weights = train_linear_regression_reg(train_X_impute_mean, train_y, 0)
val_y_pred = bias + val_X_impute_mean.dot(weights)
print("RMSE for mean imputation:", rmse(val_y, val_y_pred))

bias, weights = train_linear_regression_reg(train_X_impute_zero, train_y, 0)
val_y_pred = bias + val_X_impute_zero.dot(weights)
print("RMSE for 0 imputation:", rmse(val_y, val_y_pred))

params = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]

results = []
for param in params:
    bias, weights = train_linear_regression_reg(train_X_impute_zero, train_y, param)
    val_y_pred = bias + val_X.dot(weights)
    results.append((param, round(rmse(val_y, val_y_pred), 2)))

print(sorted(results, key=lambda x : x[1]))

seed_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
rmse_for_seeds = []
for seed_value in seed_values:

    train, val, test = divide_dataset(subset_dataframe, 0.6, 0.2, 0.2, seed_value)

    train_X = train[[c for c in columns if c != "price"]]
    train_y = np.log1p(train["price"])
    val_X = val[[c for c in columns if c != "price"]]
    val_y = np.log1p(val["price"])
    train_X_impute_zero = train_X.copy()
    train_X_impute_zero["reviews_per_month"].fillna(0, inplace=True)
    val_X_impute_zero = val_X.copy()
    val_X_impute_zero["reviews_per_month"].fillna(0, inplace=True)

    bias, weights = train_linear_regression_reg(train_X_impute_zero, train_y, 0)
    val_y_pred = bias + val_X_impute_zero.dot(weights)
    rmse_value = rmse(val_y, val_y_pred)
    print("RMSE for seed", seed_value, ":", rmse_value)
    rmse_for_seeds.append(rmse_value)

print("stdev:", round(np.std(rmse_for_seeds), 3))

## final test

train, val, test = divide_dataset(subset_dataframe, 0.6, 0.2, 0.2, 9)
print(len(train), len(val), len(test))
train = pd.concat([train, val])
train_X = train[[c for c in columns if c != "price"]]
train_y = np.log1p(train["price"])
test_X = test[[c for c in columns if c != "price"]]
test_y = np.log1p(test["price"])
train_X_impute_zero = train_X.copy()
train_X_impute_zero["reviews_per_month"].fillna(0, inplace=True)
test_X_impute_zero = test_X.copy()
test_X_impute_zero["reviews_per_month"].fillna(0, inplace=True)

bias, weights = train_linear_regression_reg(train_X_impute_zero, train_y, 0.001)
test_y_pred = bias + test_X_impute_zero.dot(weights)
print(test_y.head())
print(test_y_pred.head())
rmse_value = rmse(test_y, test_y_pred)

print(rmse_value)
