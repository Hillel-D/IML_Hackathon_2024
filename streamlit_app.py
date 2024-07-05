import base64
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from baselinepreprocess import getData, getData_updated_prepro, getData_test
import io

def svm(X_train, y_train):
    # Train an SVM model
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    return model


def svm_iteration(X_test, y_test, model):
    # Define training set sizes to iterate over
    percents_from_training = np.linspace(0.1, 1.0, 10)
    f1_scores = []
    last_y_pred = None

    for train_size in percents_from_training:
        # Create a smaller training set
        end_idx = int(len(X_test) * train_size)
        X_test_part = X_test[:end_idx]
        y_test_part = y_test[:end_idx]

        # Check if the training subset contains more than one class
        if len(np.unique(y_test)) > 1:
            # Evaluate the model
            y_pred = model.predict(X_test_part)
            f1 = f1_score(y_test_part, y_pred)
            f1_scores.append(f1)

            # Save the last predictions
            last_y_pred = y_pred
        else:
            f1_scores.append(np.nan)  # Append NaN if only one class is present

    # Plot the F1 scores
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percents_from_training * 100, y=f1_scores, mode='lines+markers', name='SVM F1 Score'))
    fig.update_layout(title='SVM F1 Score vs. Test Set Size',
                      xaxis_title='Percent of Test Set',
                      yaxis_title='F1 Score')

    # Display plot on Streamlit
    st.plotly_chart(fig)

    # Save the predictions for the last iteration
    if last_y_pred is not None:
        last_y_pred = pd.merge(X_test["unique_id"], last_y_pred)
        ground_true = pd.merge(X_test["unique_id"], y_test)
        output_df = pd.merge(last_y_pred, on='unique_id')
        # Convert DataFrame to CSV
        csv = output_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<button><a href="data:file/csv;base64,{b64}" download="match_predictions.csv">Download CSV File</a></button>'
        st.markdown(href, unsafe_allow_html=True)


def model_predict(X_test, model):
    last_y_pred = model.predict(X_test)
    print(last_y_pred)
    
    # Combine unique_id and predictions into a DataFrame
    output_df = pd.DataFrame({"unique_id": X_test["unique_id"], "match": last_y_pred})
    
    # Convert DataFrame to a byte stream (using to_csv with a StringIO buffer)
    with io.StringIO() as csv_buffer:
        output_df.to_csv(csv_buffer, index=False)
        csv = csv_buffer.getvalue().encode()  # Get the encoded bytes
        
        # Encode bytes to base64 and create the download link
        b64 = base64.b64encode(csv).decode()
        href = f'<button><a href="data:file/csv;base64,{b64}" download="match_predictions.csv">Download Prediction CSV File</a></button>'
        st.markdown(href, unsafe_allow_html=True)


def main():
    """Streamlit App"""
    st.title("BaseLine Model Analysis- Hackathon 2024")
    file_train = st.file_uploader("Choose a train data file", type="csv")
    file_test = st.file_uploader("Choose a test data file", type="csv")
    if file_train is not None and file_test is not None:
        # Read the uploaded file into a pandas DataFrame
        df_train = pd.read_csv(file_train)
        # Ensure the DataFrame is correctly passed to the data preprocessing functions
        X_train, y_train = getData(df_train)
        model = svm(X_train, y_train)
        df_test = pd.read_csv(file_test)
        # Ensure the DataFrame is correctly passed to the data preprocessing functions
        X_test = getData_test(df_test)
        st.header("Prediction")
        model_predict(X_test, model)
        #st.header("Evaluation on Full Training Set")
        #svm(X_test, y_test, model)
        #st.header("F1 Score vs. Test Set Size")
        #svm_iteration(X_test_upd, y_test_upd, model)


if __name__ == "__main__":
    main()
