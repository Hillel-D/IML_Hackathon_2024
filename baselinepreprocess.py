"""
    this file is responsible for the baseline preprocessing
"""
import numpy as np
import sklearn.model_selection as skm
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # Importing t-SNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import preprocessing

TEST_PORTION = 0.25
RANDOM_SEED = 998


def getData(df):
    filtered_df = df.dropna(subset=["match"])
    filtered_df = filtered_df.dropna()
    filtered_df = filtered_df.drop(filtered_df.filter(regex='d_').columns, axis=1)
    X, y = filtered_df.drop(["match", "has_missing_features", "professional_role", "ethnic_background", "study_field"],
                            axis=1), filtered_df.match
    return X, y


def getData(df):
    filtered_df = df.dropna()
    filtered_df = filtered_df.drop(filtered_df.filter(regex='d_').columns, axis=1)
    X = filtered_df.drop(["has_missing_features", "professional_role", "ethnic_background", "study_field"],
                            axis=1)
    return X_test


def getData_updated_prepro(df):
    df = preprocessing.preprocess_data(df)
    X, y = df.drop(["match"], axis=1), df.match
    X_train, X_test, y_train, y_test = skm.train_test_split(X, y, test_size=TEST_PORTION, random_state=RANDOM_SEED)
    print("train: ", X_train.shape, y_train.shape, "test: ",X_test.shape, y_test.shape)
    doingPCA(X, y)
    doing_tSNE(X, y)
    return X_train, X_test, y_train, y_test

def doingPCA(X, y):
    pca = PCA(n_components=10)
    pca.fit(X)
    X_transformed = pca.transform(X)

    # Plot the scatter plot of the first two principal components
    plt.figure(figsize=(8, 6))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA: First Two Principal Components')
    plt.colorbar()
    plt.show()

    # Scree plot
    explained_variance_ratio = pca.explained_variance_ratio_
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(np.arange(1, len(explained_variance_ratio) + 1))
    plt.grid()
    plt.show()

    print(X_transformed)

def doing_tSNE(X, y):
    # Standardize the data before applying t-SNE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform t-SNE with 3 components
    tsne = TSNE(n_components=3, random_state=42)  # Change n_components to 3
    X_tsne = tsne.fit_transform(X_scaled)

    # Convert y to numeric format
    y_numeric = pd.factorize(y)[0]

    # Create a DataFrame for the transformed data
    tsne_df = pd.DataFrame(data=X_tsne, columns=['tSNE Component 1', 'tSNE Component 2', 'tSNE Component 3'])
    tsne_df['Target'] = y_numeric

    # Create a 3D scatter plot using Plotly Express
    fig = px.scatter_3d(tsne_df, x='tSNE Component 1', y='tSNE Component 2', z='tSNE Component 3', color='Target',
                        labels={'tSNE Component 1': 't-SNE Component 1', 'tSNE Component 2': 't-SNE Component 2', 'tSNE Component 3': 't-SNE Component 3'},
                        title='t-SNE: First Three Components')

    fig.show()

    # Print the transformed data
    print(X_tsne)
