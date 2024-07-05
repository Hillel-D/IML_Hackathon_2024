"""
    this file is responsible for the baseline preprocessing
"""
import numpy as np
import sklearn.model_selection as skm
import pandas as pd
import plotly.express as px

TEST_PORTION = 0.25
RANDOM_SEED = 998


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
