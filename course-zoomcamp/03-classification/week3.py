import numpy as np
import pandas as pd
import sklearn
from sklearn import model_selection
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)


data = pd.read_csv("AB_NYC_2019.csv")

cols = ['neighbourhood_group', 'room_type', 'latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']

data = data[cols]
data.fillna(0, inplace=True)

print(data.head().T)

print(data['neighbourhood_group'].mode())

X = data[[c  for c in cols if c != 'price']]
y = data['price']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=42)

print(X_train.corr())

print(X_train.head().T)

above_average_train = y_train > 152
above_average_train = above_average_train.astype('int')

above_average_val = y_val > 152
above_average_val = above_average_val.astype('int')
neighborhood_group_mi = mutual_info_score(above_average_train, X_train['neighbourhood_group'])
room_type_mi = mutual_info_score(above_average_train, X_train['room_type'])

print(round(neighborhood_group_mi, 2), round(room_type_mi, 2))

dv = DictVectorizer(sparse=False)

train_dict = X_train[[c for c in cols if c != 'price']].to_dict(orient='records')
X_train_vect = dv.fit_transform(train_dict)

val_dict = X_val[[c for c in cols if c != 'price']].to_dict(orient='records')
X_val_vect = dv.transform(val_dict)

model = LogisticRegression(solver='liblinear', C=1.0, random_state=42)
model.fit(X_train_vect, above_average_train)

y_pred = model.predict_proba(X_val_vect)[:, 1]

price_decision = (y_pred >= 0.5)

train_accuracy = (above_average_val == price_decision).mean()
remove_features = ["neighbourhood_group", "room_type", "number_of_reviews", "reviews_per_month"]

for candidate in remove_features:
    without_feature_train = X_train[[c for c in X_train.columns if c != candidate]]
    without_feature_val = X_val[[c for c in X_val.columns if c != candidate]]
    model = LogisticRegression(solver='liblinear', C=1.0, random_state=42)
    dv = DictVectorizer(sparse=False)
    without_feature_train_dict = without_feature_train.to_dict(orient='records')
    without_feature_train_vect = dv.fit_transform(without_feature_train_dict)
    model.fit(without_feature_train_vect, above_average_train)

    without_feature_val_dict = without_feature_val.to_dict(orient='records')
    without_feature_val_vect = dv.transform(without_feature_val_dict)
    y_pred = model.predict_proba(without_feature_val_vect)[:, 1]
    price_decision = (y_pred >= 0.5)
    candidate_accuracy = (above_average_val == price_decision).mean()
    print(candidate, train_accuracy - candidate_accuracy)

alphas = [0, 0.01, 0.1, 1, 10]
for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train_vect, np.log1p(y_train))
    y_pred = model.predict(X_val_vect)
    rmse_value = rmse(y_val, y_pred)
    print(alpha, rmse_value)
