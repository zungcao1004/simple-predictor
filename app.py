import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

# Initialize session state variables if not already set
if "file" not in st.session_state:
    st.session_state.file = None
if "columns" not in st.session_state:
    st.session_state.columns = []
if "df" not in st.session_state:
    st.session_state.df = None

# Page layout settings
st.set_page_config(page_title="Data Analysis App", layout="wide")

st.title("Data Analysis App")

# Sidebar file uploader
select_file = st.sidebar.file_uploader("Select file")
if select_file is not None:
    st.session_state.file = select_file.getvalue()

# Button to read the file
button = st.sidebar.button("Read file")
if button:
    if st.session_state.file is not None:
        if select_file.name.endswith(".csv"):
            df = pd.read_csv(select_file)
            st.session_state.df = df
            st.session_state.columns = df.columns.tolist()
            st.write("Data preview:")
            st.write(df.head())
            st.write("Data summary:")
            st.write(df.describe())
        elif select_file.name.endswith(".xlsx"):
            df = pd.read_excel(select_file)
            st.session_state.df = df
            st.session_state.columns = df.columns.tolist()
            st.write("Data preview:")
            st.write(df.head())
            st.write("Data summary:")
            st.write(df.describe())
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")


# Function to predict with selected algorithm
def predict_with_algorithm(df, predictor_variables, target_variable, algorithm):
    X = df[predictor_variables]
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if algorithm == "Linear Regression":
        model = LinearRegression()
    elif algorithm == "Logistic Regression":
        model = LogisticRegression()
    elif algorithm == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif algorithm == "Decision Tree":
        model = DecisionTreeClassifier()
    elif algorithm == "Random Forest":
        model = RandomForestClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if algorithm == "Linear Regression":
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write("Mean Squared Error (MSE):", mse)
        st.write("Root Mean Squared Error (RMSE):", rmse)
        st.write("Mean Absolute Error (MAE):", mae)
        st.write("R-squared (R²):", r2)
    else:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        st.write("Accuracy:", accuracy)
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("F1 Score:", f1)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=y_test)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title(f'Actual vs Predicted Values for {algorithm}')
    st.pyplot(plt)

# Function to render prediction options
def predict_value():
    st.header("Predict the values")
    target_variable = st.sidebar.selectbox("Target variable", st.session_state.columns)
    predictor_variables = [col for col in st.session_state.columns if col != target_variable]
    algorithm = st.sidebar.selectbox(
        "Choose algorithm",
        [
            "Linear Regression",
            "Logistic Regression",
            "KNN",
            "Decision Tree",
            "Random Forest",
        ],
    )
    st.sidebar.subheader("Choose variables for prediction")
    selected_predictor_variables = st.sidebar.multiselect(
        "Predictor variables", predictor_variables
    )

    if st.sidebar.button("Predict"):
        if len(selected_predictor_variables) == 0:
            st.error("Please select at least one predictor variable.")
        else:
            predict_with_algorithm(
                st.session_state.df, selected_predictor_variables, target_variable, algorithm
            )

# Main function to render the app
def main():
    menu = ["Predict value"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Predict value":
        predict_value()

if __name__ == "__main__":
    main()
