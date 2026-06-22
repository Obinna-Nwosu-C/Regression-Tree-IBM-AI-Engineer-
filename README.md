Regression-Tree-IBM-AI-Engineer

NYC Taxi Tip Prediction using Regression Trees

[Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
[Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.0-orange?logo=scikit-learn)
[Pandas](https://img.shields.io/badge/Pandas-2.2.3-150458?logo=pandas)
[License](https://img.shields.io/badge/License-MIT-green)

Project Overview
This project applies Decision Tree Regression** to predict the continuous tip amount (`tip_amount`) for NYC yellow taxi trips. Using real-world data from the Taxi & Limousine Commission (TLC), the model analyzes trip characteristics to estimate passenger tipping behavior. 

Unlike classification trees that output a category, this Regression Tree outputs a continuous dollar value by calculating the average tip of historical trips that share similar characteristics (minimizing variance using the Mean Squared Error criterion).

The Dataset
The dataset is a subset of the publicly available [NYC TLC Yellow Taxi Trip Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page). 
Target Variable: `tip_amount` (Continuous numerical value)
Key Features: `fare_amount`, `trip_distance`, `tolls_amount`, `total_amount`, etc.
Dropped Features: Categorical/ID variables like `VendorID`, `payment_type`, and `store_and_fwd_flag` were removed as they do not contribute to the mathematical magnitude of a tip.

Methodology & Workflow

1. Data Preprocessing
Feature Selection: Removed non-numeric and irrelevant ID columns to reduce noise.
Normalization: Applied L1 Normalization (`normalize(X, axis=1, norm='l1')`) to the feature matrix. This scales the features so that variables with larger numerical ranges (like total fare) don't disproportionately bias the tree's splitting criteria compared to smaller variables.
Train/Test Split: Partitioned the data into 70% training and 30% testing sets (`random_state=42`).

2. Model Training & Evaluation
Trained a `DecisionTreeRegressor` using the `squared_error` (MSE) criterion.
Evaluated model performance using Mean Squared Error (MSE) and the **Coefficient of Determination (R^2).

3. Hyperparameter Tuning (The Overfitting Lesson)
A core focus of this project was analyzing how tree complexity (`max_depth`) impacts generalization:

| Max Depth | MSE Score | R^2 Score | Observation |
| 8 (Baseline) | 24.555 | 0.028 | Standard performance. |
| 12 (Too Deep) | 26.875 | -0.064 | Overfitting - The model memorized training noise, resulting in a negative $R^2$ (worse than predicting the mean). |
| 4 (Optimal) | 24.468 | 0.031 | Best Generalization. Restricting the tree forced it to learn broader, more reliable patterns. |

Key Insights & Feature Correlation
By analyzing the correlation between the target variable and the input features, the top 3 strongest predictors of tip amount are:
1. Fare Amount (Correlation: ~0.20)
2. Tolls Amount (Correlation: ~0.11)
3. Trip Distance (Correlation: ~0.10)

Real-World Takeaway: While the model successfully identifies the mathematical drivers of a tip, the overall $R^2$ remains relatively low (~0.03). This is highly expected in behavioral data science; human tipping habits are influenced by subjective factors (mood, service quality, cash vs. card) that are not captured in the TLC trip metadata.

