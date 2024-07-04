import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from baselinepreprocess import getData, getData_updated_prepro


def svm(filename):
    # Get preprocessed data
    X_train, X_test, y_train, y_test = getData(filename)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train an SVM model
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    test_f1_score = f1_score(y_test, y_pred)

    # Display results on Streamlit
    st.write("Test F1 Score:", test_f1_score)

    # Create a DataFrame with the predictions (optional)
    # output = pd.DataFrame({'unique_id': X_test.index, 'match': y_pred})
    # output.to_csv('match_predictions.csv', index=False)


def svm_iteration(filename):
    # Get preprocessed data
    X_train, X_test, y_train, y_test = getData_updated_prepro(filename)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define training set sizes to iterate over
    percents_from_training = np.linspace(0.1, 1.0, 10)
    f1_scores = []

    for train_size in percents_from_training:
        # Create a smaller training set
        end_idx = int(len(X_train_scaled) * train_size)
        X_train_part = X_train_scaled[:end_idx]
        y_train_part = y_train[:end_idx]

        # Check if the training subset contains more than one class
        if len(np.unique(y_train_part)) > 1:
            # Train an SVM model
            model = SVC(kernel='linear', random_state=42)
            model.fit(X_train_part, y_train_part)

            # Evaluate the model
            y_pred = model.predict(X_test_scaled)
            f1 = f1_score(y_test, y_pred)
            f1_scores.append(f1)
        else:
            f1_scores.append(np.nan)  # Append NaN if only one class is present

    # Plot the F1 scores
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percents_from_training * 100, y=f1_scores, mode='lines+markers', name='SVM F1 Score'))
    fig.update_layout(title='SVM F1 Score vs. Training Set Size',
                      xaxis_title='Percent of Training Set',
                      yaxis_title='F1 Score')

    # Display plot on Streamlit
    st.plotly_chart(fig)


def main():
    """Streamlit App"""
    st.title("BaseLine Model Analysis")
    file_uploader = st.file_uploader("Choose a preprocessed data file", type="csv")

    if file_uploader is not None:
        filename = file_uploader.name
        st.header("Evaluation on Full Training Set")
        svm(filename)
        st.header("F1 Score vs. Training Set Size")
        svm_iteration(filename)


if __name__ == "__main__":
    main()
