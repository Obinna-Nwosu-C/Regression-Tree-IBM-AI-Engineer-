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

Further Explanation of the Exercise

The Problem: 
Unlike your previous project where we predicted which drug a patient would take (Classification), this project tackles a Regression problem: predicting a continuous number. Specifically, we want to predict the exact dollar amount a passenger will tip a NYC taxi driver based on the details of their trip. 

The Approach: 
I used a Decision Tree Regressor. While a classification tree ends its leaves with a "majority vote," a regression tree ends its leaves with the average value of the tips for all the trips that landed in that specific leaf. To build the tree, it uses Mean Squared Error (MSE) to find the splits that group similar tip amounts together.

The Data & Preprocessing:
I used the real-world NYC TLC (Taxi & Limousine Commission) yellow taxi dataset. 

    Cleaning: I dropped irrelevant categorical and ID columns (like VendorID and payment_type) that don't mathematically help predict the tip size.
    Scaling: I applied L1 Normalization to the feature matrix. This ensures that features with massive numbers (like total fare) don't unfairly dominate features with smaller numbers during the splitting process.

The Results & (Hyperparameter Tuning):
The most valuable part of this exercise was experimenting with the tree's max_depth (how many questions it's allowed to ask) and observing the MSE and R^2 (R-squared) scores.
    Depth 8 (Baseline): The model achieved an MSE of ~24.5 and an R^2 of 0.028. 
    Depth 12 (Overfitting): I thought a deeper tree would be better, so I increased the depth to 12. Surprisingly, the MSE increased to 26.8, and the R^2 dropped to a negative number (-0.064). A negative R^2 means the model is literally performing worse than if it just guessed the average tip for every single person! The tree had memorized the training data (overfitting) and failed to generalize.
    Depth 4 (The Sweet Spot): By restricting the tree to a depth of 4, the MSE dropped to 24.4 and the R^2 improved to 0.031. The simpler tree generalized much better to unseen data.

Key Insight:
I also ran a correlation analysis and found that the Fare Amount, Tolls Amount, and Trip Distance are the strongest mathematical predictors of how much a passenger will tip. However, because human tipping behavior is highly subjective and volatile, the overall R^2 remains relatively low—a very realistic outcome in real-world behavioral data science.

Based on the notebook and the nature of the model trained, there is no single, fixed exact dollar amount that the model predicts for every passenger. 
Because this is a Regression Tree, the predicted tip is dynamic. It changes on a case-by-case basis depending entirely on the specific details of that individual trip.
Here is how the model determines the "exact" amount for a specific passenger:

1. It looks at the specific trip details
When a new trip comes in, the model looks at the specific numerical values for the features you kept in the dataset, primarily:

    Fare Amount (The strongest predictor)
    Tolls Amount
    Trip Distance

2. It travels down the tree
The model uses those specific numbers to answer a series of Yes/No questions (the splits) that it learned during training. For example: "Is the fare amount greater than $50?" -> "Are there tolls?" -> "Is the distance greater than 10 miles?"

3. It outputs the Leaf Node Average
Once the trip's details filter all the way down to a final Leaf Node, the model looks at all the historical trips in your training data that ended up in that exact same leaf. It calculates the mathematical average (mean) of their actual tips, and outputs that as the exact predicted dollar amount.

Examples from your dataset:
If you look at the raw data provided in your notebook, the actual historical tips (tip_amount) vary wildly based on the trip. In just the first few rows of your data, the tips are:

    $16.54 (for a trip with a $70.00 fare and $6.94 in tolls)
    $12.00 (for a similar fare/toll setup)
    $5.00 (for a trip with $0.00 in tolls)
    $4.13

The Reality Check (MSE & R^2)
It is also important to remember the evaluation metrics from your exercise. The optimal model (Depth 4) yielded an R^2 score of roughly 0.03 and an MSE of ~24.4. 
This means that while the model will output an exact dollar amount (e.g., "$8.42"), human tipping behavior is so subjective and volatile that this exact prediction will likely have a margin of error of several dollars. The model provides the best statistical estimate based on the fare and distance, but it cannot perfectly predict the exact change a specific passenger will decide to leave.

Course Participant/Code Executor: Obinna Nwosu C, Author: Jeff Grossman, Other Contributor(s): Abhishek Gagneja, © IBM Corporation. All rights reserved.
