import base64
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from io import StringIO

# ---- Constants ----
STRING_COLS = ['professional_role', 'ethnic_background', 'ethnic_background_o', 'study_field', 'shared_ethnicity']
COLS_TO_DROP = ['unique_id', 'has_missing_features', 'wave']
PREFIX = "b'"
SUFFIX = "'"
SPLITTER = "/"
TEST_PORTION = 0.25
RANDOM_SEED = 998
TASK2_COLS = ['creativity_important', 'ambition_important']


# ---- handle data ----

def read_data(file_name):
    df = pd.read_csv(file_name)
    return df


def drop_match(df: pd.DataFrame):
    return df.drop(columns=['match'], axis=1)


def split_supervised_unsupervised(df: pd.DataFrame):
    supervised = df[df['match'].isin([0, 1])]
    unsupervised = df[df['match'].isna()]
    return supervised, unsupervised


# ---- help functions for preprocess on supervised ----
def drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    """A function that drops all columns that hold binned variables."""
    cols_to_drop = df.filter(regex='^d_').columns
    df_dropped = df.drop(columns=cols_to_drop)
    df_dropped = df_dropped.drop(columns=COLS_TO_DROP)
    return df_dropped


def get_dummies(df: pd.DataFrame) -> pd.DataFrame:
    data = pd.get_dummies(df, columns=['professional_role', 'study_field'])
    return data


def change_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in STRING_COLS:
        df[col] = df[col].str.lstrip(PREFIX).str.rstrip(SUFFIX)
    return df


def split_ethnicity(df):
    ethnicity_bank = set()
    ethnic_cols = ['ethnic_background', 'ethnic_background_o']
    prefix_dict = {'ethnic_background': 'self_', 'ethnic_background_o': 'other_'}
    for col in ethnic_cols:
        df[col] = df[col].str.split(SPLITTER)
        for entry in df[col].dropna():
            for entry_ethnicity in entry:
                ethnicity_bank.add(entry_ethnicity)
        for ethnicity in ethnicity_bank:
            df[prefix_dict[col] + ethnicity] = df[col].apply(lambda x: 1 if ethnicity in x else 0)
    df = df.drop(columns=ethnic_cols)
    return df


def remove_outlier(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    df = df[df['creativity_important'].between(5, 31)]
    df = df[df['ambition_important'] < 22]
    return df


# ---- combine preprocess ----
def preprocess_train(X_train: pd.DataFrame, y_train: pd.Series):
    X, y = X_train, y_train
    X = X.dropna().drop_duplicates()
    # X = remove_outlier(X)
    y = y.loc[X.index]
    return X, y

def getData_Task2(df: pd.DataFrame):
    df = drop_cols(df)
    df = change_string_columns(df)
    df = split_ethnicity(df)
    df = get_dummies(df)
    df = df.astype(float)
    df = drop_match(df)
    df = df.fillna(0)

    X, y = (df.drop(columns=TASK2_COLS),
            df[TASK2_COLS])

    X_train, y_train = preprocess_train(X, y)
    
    y_train_creativity_important, y_train_ambition_important = y_train[TASK2_COLS[0]], y_train[
        TASK2_COLS[1]]

    return X_train, (y_train_creativity_important, y_train_ambition_important)

def getData_test_Task2(df: pd.DataFrame, additional_cols=None):
    df = drop_cols(df)
    df = change_string_columns(df)
    df = split_ethnicity(df)
    df = get_dummies(df)
    df = df.astype(float)
    df = df.fillna(0)
    df.dropna().drop_duplicates()
    if additional_cols is not None:
        missing_cols = [col for col in additional_cols if col not in df.columns]
        if missing_cols:
            df = df.reindex(columns=df.columns.tolist() + missing_cols)
            df[missing_cols] = 0
    return df

def getData(df):
    filtered_df = df.dropna(subset=["match"])
    filtered_df = filtered_df.dropna()
    filtered_df = filtered_df.drop(filtered_df.filter(regex='d_').columns, axis=1)
    X, y = filtered_df.drop(["match", "has_missing_features", "professional_role", "ethnic_background", "study_field"],
                            axis=1), filtered_df.match
    return X, y


def getData_test(df):
    filtered_df = df.dropna()
    filtered_df = filtered_df.drop(filtered_df.filter(regex='d_').columns, axis=1)
    X = filtered_df.drop(["has_missing_features", "professional_role", "ethnic_background", "study_field"],
                            axis=1)
    return X


def svm(X_train, y_train):
    # Train an SVM model
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    return model


def svm_model_predict(X_test, model):
    last_y_pred = model.predict(X_test)
    output_df = pd.DataFrame({"unique_id": X_test["unique_id"], "match": last_y_pred})

    with StringIO() as csv_buffer:
        output_df.to_csv(csv_buffer, index=False)
        csv = csv_buffer.getvalue().encode()  # Get the encoded bytes
        b64 = base64.b64encode(csv).decode()
        href = f'<button><a href="data:file/csv;base64,{b64}" download="match_predictions.csv">Download Prediction CSV File</a></button>'
    return href


def tree(X_train, y_train, X_test):
    results = {}
    for index, col in enumerate(TASK2_COLS):
        y_train_col = y_train[index].round()

        model = DecisionTreeClassifier(random_state=998, max_depth=25)
        model.fit(X_train, y_train_col)

        y_pred = model.predict(X_test)
        
        output_df = pd.DataFrame({"unique_id": X_test["unique_id"], col: y_pred})
        with StringIO() as csv_buffer:
            output_df.to_csv(csv_buffer, index=False)
            csv = csv_buffer.getvalue().encode()  # Get the encoded bytes
            b64 = base64.b64encode(csv).decode()
            href = f'<button><a href="data:file/csv;base64,{b64}" download="{col}_predictions.csv">Download Predictions CSV File</a></button>'
        results[f"{col}_href"] = href

    return results
