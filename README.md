# COMPAS Recidivism Prediction & Fairness Analysis

This project analyzes the COMPAS dataset to predict criminal recidivism while evaluating potential algorithmic bias and fairness issues.

The project follows a complete end-to-end Data Science workflow:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Modeling (The "01_Notebook - test" folder contains all the models that have been tested, whilst the "01_Notebook - final" folder contains the final model)
- Dashboard Data Preparation (Link to dashboard: https://lookerstudio.google.com/reporting/767c057b-92a8-4ca0-8d44-8d04b8593a3f)

The goal is not only to build predictive models but also to evaluate whether these models introduce or amplify bias across demographic groups.

## Project Objective

The main objectives of this project are:

Clean and standardize raw criminal justice data
Explore patterns related to recidivism
Engineer relevant predictive features
Train multiple classification models
Evaluate model performance
Analyze fairness metrics
Prepare datasets for dashboard visualization

This project focuses on the ethical challenges of predictive systems used in high-impact environments such as criminal justice.

## Libraries used
pandas, numpy, matplotlib, seaborn, plotly, missingno, rapidfuzz, scikit-learn and fairlearn.


## Final Insight

The key to this project is the bias found in the race variable; our model has succeeded in eliminating this bias for each racial group, whilst also improving the detection of repeat offenders overall compared to the model provided by ProPublica.

## General comparison of models

<img width="1025" height="374" alt="image" src="https://github.com/user-attachments/assets/c81f3aad-d8a0-48ba-b131-346573d538cc" />
