**Industrial Copper Modeling Project**

**Project Overview**
The Industrial Copper Modeling Project leverages advanced machine learning techniques to address challenges in the copper industry. The primary focus is on analyzing sales and pricing data to develop predictive models that enhance decision-making processes. These models aim to tackle issues such as data skewness, noise, and lead classification, providing actionable insights into pricing strategies and customer behavior.

**Problem Statement**
The copper industry faces significant challenges in analyzing sales and pricing data due to:

**Data Skewness**: Affects the accuracy of predictive models.
Noise and Outliers: Introduce variability, reducing the reliability of manual or basic statistical methods.
Manual Lead Classification: Time-intensive and error-prone processes for evaluating leads (status: WON or LOST).
To address these, we developed:

**A Regression Model** to predict the continuous variable, selling_price.
**A Classification Model** to evaluate leads based on their likelihood of conversion (status: WON or LOST).

**Data Description**
The dataset used for this project contains the following columns:

id: Unique identifier for each transaction.
item_date: Date of the transaction.
quantity_tons: Quantity sold in tons.
customer: Customer identifier.
country: Customer's country.
status: Transaction status (WON or LOST).
item_type: Category of items sold.
application: Specific use of items.
thickness, width: Physical dimensions of the items.
material_ref, product_ref: Material and product references.
delivery_date: Expected delivery date.
selling_price: Price at which items are sold.


**Why This Model Was Selected**
Challenges Addressed by Machine Learning
Handling Skewness: Models normalize skewed data to improve accuracy.
Robust to Noise and Outliers: Techniques like Isolation Forest and Random Forest handle variability effectively.
Enhanced Lead Classification: Machine learning provides precise evaluation of lead conversion likelihood.
Superiority Over Alternatives
Manual Approaches: Time-intensive and error-prone.
Basic Statistical Models: Fail to capture complex, non-linear relationships.
Machine Learning Models: Excel in robustness and predictive accuracy.

**Data Analysis Workflow**

**1. Data Preprocessing**
Handling Missing Values: Applied mean/median imputation.
Outlier Treatment: Used IQR and Isolation Forest methods.
Skewness Treatment: Log transformations reduced skewness in continuous variables like selling_price.
Categorical Encoding: Converted variables (e.g., customer, country, application) using one-hot encoding.

**2. Exploratory Data Analysis (EDA)**
EDA was conducted using various graphical methods:

Boxplots: Highlighted outliers in selling_price, informing preprocessing strategies.
Scatter Plots: Demonstrated relationships between features like quantity_tons and selling_price.
Violin Plots: Illustrated differences in selling_price distributions across categories like application.
Histograms: Visualized frequency distributions to address skewness.
Pairplots: Detected feature correlations, guiding feature engineering.

**Model Selection and Rationale**
**Explored Algorithms**
Linear Regression: Simple but unsuitable for non-linear data relationships.
Decision Tree Regressor/Classifier: Effective but prone to overfitting.
Random Forest Regressor/Classifier (Selected): Robust, interpretable, and highly accurate.
XGBoost: High-performing but overly complex for this dataset.
Final Models Used
Random Forest Regressor: For predicting selling_price.
Random Forest Classifier: For classifying lead status (WON/LOST).

**Why Random Forest?**
Robust against noise, outliers, and skewness.
Handles both linear and non-linear relationships.
Provides feature importance insights for transparency.
Demonstrated superior performance metrics compared to alternatives.
Results
Regression Model (Selling Price Prediction)
Algorithm: Random Forest Regressor.
Performance Metrics:
R² Score: 0.92
Mean Squared Error (MSE): Low, indicating high predictive accuracy.

**Impact**: Enhanced pricing accuracy, leading to optimal pricing strategies.
Classification Model (Lead Status Prediction)
Algorithm: Random Forest Classifier.
Performance Metrics:
Precision: 89%
Recall: 86%
F1-Score: 87%
**Impact**: Improved lead prioritization, focusing efforts on high-conversion leads.

**Deployment**
The models were deployed using Streamlit, providing stakeholders with:

Interactive Visualizations: EDA insights are available in an intuitive interface.
Real-Time Predictions:
Predict selling_price based on user inputs.
Classify lead status (WON/LOST).
Insights Gained
Outliers Impact Pricing Decisions:
Outliers in selling_price required treatment to improve model reliability.

**Feature Importance:**
Significant predictors like quantity_tons and application were identified.
**Skewness Treatment:**
Log transformations reduced skewness, enhancing model performance.
**Customer Behavior:**
Identified key customers driving major transactions, aiding segmentation.

**Future Scope**
Advanced Algorithms: Explore Gradient Boosting and Neural Networks for further accuracy.
Real-Time Data Integration: Enable live updates for predictions and decision-making.
Expanded Application: Extend the system to include inventory management and sales forecasting.
Enhanced Explainability: Use SHAP values to improve stakeholder understanding of model outputs.
Automated Model Retraining: Adapt to market changes with periodic retraining.

**Technical Deliverables**
Code: Python implementation using Pandas, Scikit-learn, and Streamlit.
Repository: Public GitHub repository with modular, PEP 8-compliant code.
Documentation: Comprehensive README explaining project workflow and steps for execution.
Application: Interactive Streamlit app for predictions and visualizations.

**Project Workflow**
Data Preprocessing:

Handled missing values, outliers, and skewness.
Encoded categorical variables for machine learning compatibility.
EDA:

Conducted detailed visualizations using Seaborn and Matplotlib.
Model Building:

Trained Random Forest models for regression and classification.
Optimized using cross-validation and grid search.
Deployment:

Built an interactive web application using Streamlit.

**Learning Outcomes**
Proficiency in Python libraries (Pandas, NumPy, Scikit-learn, Streamlit).
Expertise in data preprocessing, EDA, and machine learning.
Deployment of interactive applications.
