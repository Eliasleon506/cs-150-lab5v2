# cs-150-lab5

### Classification Problem: Airline Passenger Satisfaction

#### Overview
My project is about predicting passenger satisfaction based on in-flight experience and flight-related features. The goal is to build a binary classification model that categorizes passengers as either "Satisfied" or "Not Satisfied".

#### Dataset
The dataset I used is the Airline Passenger Satisfaction dataset from Kaggle.
#### How
To make the prediction model I converted the following to a numeric value using one-hot encoding:
* Gender
* Customer Type
* Type of Travel
* Class

These features are used to train and visualize a binary SVM classifier to predict satisfaction.

#### Target Variable
- **Satisfied** (positive class)
- **Neutral or Dissatisfied** (negative class)
