import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from preprocessing_and_models import (
    getData, getData_test, getData_Task2, getData_test_Task2,
    svm, svm_model_predict, tree
)

def main():
    """Streamlit App"""
    st.title("BaseLine Models Analysis- Hackathon 2024")

    # Task 1: SVM
    st.header("Task 1: SVM Model")
    file_train_task1 = st.file_uploader("Choose a Task 1 train data file", type="csv")
    file_test_task1 = st.file_uploader("Choose a Task 1 test data file", type="csv")
    if file_train_task1 is not None and file_test_task1 is not None:
        # Read the uploaded file into a pandas DataFrame
        df_train = pd.read_csv(file_train_task1)
        # Ensure the DataFrame is correctly passed to the data preprocessing functions
        X_train, y_train = getData(df_train)
        model = svm(X_train, y_train)
        df_test = pd.read_csv(file_test_task1)
        # Ensure the DataFrame is correctly passed to the data preprocessing functions
        X_test = getData_test(df_test)
        st.header("Prediction")
        href = model_predict(X_test, model)
        st.markdown(href, unsafe_allow_html=True)

    # Task 2: Decision Tree
    st.header("Task 2: Decision Tree Model")
    file_train_task2 = st.file_uploader("Choose a Task 2 train data file", type="csv")
    file_test_task2 = st.file_uploader("Choose a Task 2 test data file", type="csv")
    if file_train_task2 is not None and file_test_task2 is not None:
        # Read the uploaded file into a pandas DataFrame
        df_train = pd.read_csv(file_train_task2)
        # Ensure the DataFrame is correctly passed to the data preprocessing functions
        X_train, y_train = getData_Task2(df_train)
        df_test = pd.read_csv(file_test_task2)
        # Ensure the DataFrame is correctly passed to the data preprocessing functions
        X_test = getData_test_Task2(df_test)
        st.header("Prediction")
        results = tree(X_train, y_train, X_test)
        for col in results:
            if "_href" in col:
                st.markdown(results[col], unsafe_allow_html=True)
            else:
                st.write(f"Mean squared error for {col}: {results[col]}")

if __name__ == "__main__":
    main()
