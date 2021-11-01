import os, sys, pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge

TRACKS_FILENAME = "spotify_tracks.csv"
LYRICS_FILENAME = "lyrics_features.csv"

COLUMNS = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness",
          "liveness", "loudness", "speechiness", "tempo", "time_signature", "valence",
          "mean_syllables_word", "mean_words_sentence", "n_sentences", "n_words",
           "vocabulary_wealth", "popularity"]
TARGET = "popularity"
TRAINING_FEATURES = [f for f in COLUMNS if f != TARGET]
MODEL_FILENAME = "ridge_regressor_polynomial_alpha_1e-1.bin"
DICT_VECTORIZER_FILENAME = "spotify_dict_vectorizer.bin"

def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

def create_dataframe(directory="./"):
    try:
        tracks_df = pd.read_csv(os.path.join(directory, TRACKS_FILENAME))
        lyrics_df = pd.read_csv(os.path.join(directory, LYRICS_FILENAME))
    except FileNotFoundError:
        sys.exit("""Spotify CSV files were not found. Please make sure that they have been downloaded from this page: \
https://www.kaggle.com/saurabhshahane/spotgen-music-dataset and that the CSV files are in the same \
directory as the script.""")

    tracks_and_lyrics_df = tracks_df.merge(lyrics_df, how='left', left_on='id', right_on='track_id')
    subset_df = tracks_and_lyrics_df[COLUMNS]

    return subset_df

def train_val_test_split(dataframe):
    train_val, test = train_test_split(dataframe, test_size=0.2, random_state=1)

    train_val = train_val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train_val_y = train_val["popularity"]
    test_y = test["popularity"]

    # make copies of the data - maybe we don't need this...
    train_val_X = train_val.copy()
    test_X = test.copy()

    del train_val_X["popularity"]
    del test_X["popularity"]

    return (train_val_X, test_X, train_val_y, test_y)

def prepare_input(df, dv=None, polynomial=False, scale_features=True):
    # input na with 0
    df_imputed = df.fillna(0)

    # feature scaling
    df_scaled = df_imputed

    if scale_features:
        scaler = RobustScaler()
        df_scaled = scaler.fit_transform(df_imputed)
        df_scaled = pd.DataFrame(df_scaled, columns=TRAINING_FEATURES)

    df_polynomial = df_scaled
    if polynomial:
        polynomial_features = PolynomialFeatures(degree = 2, include_bias = False)
        df_polynomial = polynomial_features.fit_transform(df_scaled)
        df_polynomial = pd.DataFrame(df_polynomial)

    df_dict = df_polynomial.to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer()
        df_vectorized = dv.fit_transform(df_dict)
        return df_vectorized, dv
    else:
        df_vectorized = dv.transform(df_dict)

    return df_vectorized

def train_ridge_regressor(train_X, train_y, polynomial=False, c=1):
    rr = Ridge(alpha=c)
    train_X_prepared, dv = prepare_input(train_X, None, polynomial)
    rr.fit(train_X_prepared, train_y)

    return rr, dv

def test_model(X, y, model, dict_vectorizer):
    X_prepared = prepare_input(X, dict_vectorizer, True, True)
    y_pred = model.predict(X_prepared)
    test_rmse = rmse(y, y_pred)

    print("RMSE for Linear Regression on Test Data:", test_rmse)

def save_bins(model, dict_vectorizer):
    model_bin = open(MODEL_FILENAME, 'wb')
    dv_bin = open(DICT_VECTORIZER_FILENAME, 'wb')

    pickle.dump(rr, model_bin)
    pickle.dump(dv, dv_bin)

    model_bin.close()
    dv_bin.close()

if __name__ == "__main__":
    df = create_dataframe()
    train_val_X, test_X, train_val_y, test_y = train_val_test_split(df)
    
    rr, dv = train_ridge_regressor(train_val_X, train_val_y, True, 0.1)
    test_model(test_X, test_y, rr, dv)

    save_bins(rr, dv)
