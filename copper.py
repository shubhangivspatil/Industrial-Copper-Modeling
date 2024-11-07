import pandas as pd
import joblib
import streamlit as st
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import skew

# Define paths for models and data
model_path = r"C:\Users\User\Desktop\jupiter_files"  # Path for saving models
file_path = r'D:/Final_Projects/New_Cleaned_Copper_modelling_project.csv'  # Path to the cleaned dataset

# Load and prepare the dataset
def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path)
    # Define features and target variables
    X = data[['quantity tons', 'customer', 'country', 'application', 'thickness', 'width']]
    y_price = data['selling_price']
    y_status = data['status_Won']  # Assuming status_Won is the target for classification

    # Convert categorical variables to numerical (if necessary)
    X = pd.get_dummies(X)
    return X, y_price, y_status, data

# Train models function
def train_models(X, y_price, y_status):
    # Split the data into training and testing sets
    X_train, X_test, y_train_price, y_test_price = train_test_split(X, y_price, test_size=0.2, random_state=42)
    _, _, y_train_status, y_test_status = train_test_split(X, y_status, test_size=0.2, random_state=42)

    # Train regression model for selling_price prediction
    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train_price)
    joblib.dump(regressor, os.path.join(model_path, 'best_rf_regressor.pkl'))

    # Train classification model for status_Won prediction
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train_status)
    joblib.dump(classifier, os.path.join(model_path, 'best_rf_classifier.pkl'))

# Load models function
@st.cache_resource
def load_models():
    try:
        regressor = joblib.load(os.path.join(model_path, 'best_rf_regressor.pkl'))
        classifier = joblib.load(os.path.join(model_path, 'best_rf_classifier.pkl'))
        return regressor, classifier
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        return None, None

# Load data function
@st.cache_data
def load_data():
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        return None

# Advanced EDA Function
def advanced_eda(data):
    st.title("Advanced Exploratory Data Analysis (EDA)")

    # Selling Price Distribution before skew treatment
    st.subheader("Selling Price Distribution Before Treatment")
    plt.figure(figsize=(10,6))
    sns.histplot(data['selling_price'], kde=True, color='blue', bins=30)
    st.pyplot()

    # Boxplot for Outliers Detection
    st.subheader("Outliers Detection - Selling Price")
    plt.figure(figsize=(10,6))
    sns.boxplot(data['selling_price'], color='red')
    st.pyplot()

    # Check skewness of 'selling_price' before treatment
    skewness_before = skew(data['selling_price'])
    st.write(f"Skewness of Selling Price before treatment: {skewness_before:.2f}")

    # Log transformation to treat skewness
    data['log_selling_price'] = np.log1p(data['selling_price'])
    skewness_after = skew(data['log_selling_price'])
    st.write(f"Skewness of Selling Price after log transformation: {skewness_after:.2f}")

    # Selling Price Distribution after skew treatment
    st.subheader("Selling Price Distribution After Log Transformation")
    plt.figure(figsize=(10,6))
    sns.histplot(data['log_selling_price'], kde=True, color='green', bins=30)
    st.pyplot()

    # Violin Plot for Distribution (before and after treatment)
    st.subheader("Violin Plot: Selling Price Before and After Transformation")
    plt.figure(figsize=(12,6))
    sns.violinplot(data=data[['selling_price', 'log_selling_price']], palette="Set2")
    st.pyplot()

    # Pairplot to visualize relationships between features
    st.subheader("Pairplot of Selected Features")
    sns.pairplot(data[['quantity tons', 'thickness', 'width', 'selling_price']], hue='selling_price')
    st.pyplot()

# Main Streamlit app function
def main():
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Select a page", ["Home", "Predictive Analysis", "Retrain Models", "Advanced EDA"])

    # Load and prepare data
    X, y_price, y_status, data = load_and_prepare_data(file_path)

    if page == "Home":
        st.title("Welcome to the Industrial Copper Modeling App")
        st.write("""
        This application allows you to predict selling prices and classify status (Won/Lost) based on various features.
        
        **Problem Statement:**
        The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, feature scaling, and outlier detection, and leveraging algorithms that are robust to skewed and noisy data.

        **Lead Classification:**
        The copper industry faces challenges in capturing leads. The status of a lead can be classified as 'WON' (Success) or 'LOST' (Failure) based on the likelihood of becoming a customer.

        **Skills Learned:**
        Python scripting, Data Preprocessing, EDA, Streamlit

        **GitHub Link:**
        [Shubhangi Patil GitHub](https://github.com/shubhangivspatil/Industrial-Copper-Modeling)  
        """)

    elif page == "Predictive Analysis":
        # Load models for prediction
        regressor, classifier = load_models()
        
        if regressor is None or classifier is None:
            st.error("Models are not loaded. Please retrain models first.")
            return

        # Input form for predictions
        st.sidebar.header("User Input Features")
        columns = ['quantity tons', 'customer', 'country', 'application', 'thickness', 'width']
        
        # Collect input data from the user
        input_data = {}
        for col in columns:
            if col in ['customer', 'country', 'application']:  # Assuming these are categorical features
                input_data[col] = st.sidebar.selectbox(f"Select {col}", options=data[col].unique())
            else:
                input_data[col] = st.sidebar.number_input(f"Enter {col}", value=0.0)
        
        # Convert user input to DataFrame and one-hot encode
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)  # Match train columns
        
        # Predict Selling Price
        if st.button("Predict Selling Price"):
            try:
                selling_price_pred = regressor.predict(input_df)[0]
                st.write(f"### Predicted Selling Price: ${selling_price_pred:,.2f}")  # Format as dollars
            except Exception as e:
                st.error(f"Error predicting selling price: {e}")

        # Predict Status (using status_Won)
        if st.button("Predict Status (Won/Lost)"):
            try:
                status_pred = classifier.predict(input_df)[0]
                status = "WON" if status_pred == 1 else "LOST"
                st.write(f"### Predicted Status: {status}")
            except Exception as e:
                st.error(f"Error predicting status: {e}")

    elif page == "Retrain Models":
        st.title("Retrain Models")
        st.write("Click the button below to retrain models with the latest data.")
        
        if st.button("Retrain Models"):
            # Load and preprocess data
            X, y_price, y_status, data = load_and_prepare_data(file_path)
            # Train and save the models
            train_models(X, y_price, y_status)
            st.success("Models retrained and saved successfully.")
            st.write("Reload the **Predictive Analysis** page to use the updated models.")

    elif page == "Advanced EDA":
        # Perform and display advanced EDA
        advanced_eda(data)

if __name__ == "__main__":
    main()
