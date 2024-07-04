import pandas as pd

#### CONSTANTS ####
STRING_COLS = ['professional_role', 'ethnic_background', 'ethnic_background_o', 'study_field', 'shared_ethnicity']
COLS_TO_DROP = ['unique_id', 'has_missing_features', 'wave',]
PREFIX = "b'"
SUFFIX = "'"
SPLITTER = "/"


#### CODE ####

def cool_facts(df):
    missing_values_rows = df["has_missing_features"].unique()
    print(missing_values_rows)


def drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    """ A function that drops all columns that hold binned variables.
    @:param df: our dataframe."""
    cols_to_drop = df.filter(regex='^d_').columns
    df_dropped = df.drop(columns=cols_to_drop)
    df_dropped = df_dropped.drop(columns=COLS_TO_DROP)
    return df_dropped


def get_dummies(df: pd.DataFrame) -> pd.DataFrame:
    data = pd.get_dummies(df, columns=['professional_role'])
    data = pd.get_dummies(data, columns=['study_field'])
    return data


def change_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in STRING_COLS:
        df[col] = df[col].str.lstrip(PREFIX)
        df[col] = df[col].str.rstrip(SUFFIX)
    return df


def split_ethnicity(df):
    ethnicity_bank = set()
    ethnic_cols = ['ethnic_background', 'ethnic_background_o']
    prefix_dict = {'ethnic_background': 'self_', 'ethnic_background_o': 'other_'}
    for col in ethnic_cols:
        df[col] = df[col].str.split(SPLITTER)
        for entry in df[col]:
            for entry_ethnicity in entry:
                ethnicity_bank.add(entry_ethnicity)
        for ethnicity in ethnicity_bank:
            df[prefix_dict[col] + ethnicity] = df[col].apply(lambda x: 1 if ethnicity in x else 0)
    df = df.drop(columns=ethnic_cols)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates()
    df = get_dummies(split_ethnicity(change_string_columns(drop_cols(df))))
    return df.astype(float)


def preprocess_data_without_match(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(['match'], axis=1)
    df = df.dropna()
    df = df.drop_duplicates()
    df = get_dummies(split_ethnicity(change_string_columns(drop_cols(df))))
    return df.astype(float)

def remove_outlier(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


