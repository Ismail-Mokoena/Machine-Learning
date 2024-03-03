# Bank Customer Churn Prediction

This project aims to predict whether a bank customer will stay or leave based on their information using an artificial neural network (ANN) trained on bank data.

## Overview

Customer churn, the rate at which customers leave a product or service, is a critical metric for businesses, especially in industries like banking where customer retention is key to profitability. By leveraging machine learning techniques, particularly artificial neural networks, we can build models to predict churn and take proactive measures to retain customers.

In this project, we have developed an ANN model trained on a dataset containing various attributes of bank customers, such as their demographics, transaction history, and customer service interactions. The model analyzes this data to predict the likelihood of a customer leaving the bank.

## Dataset

The dataset used for training and evaluation contains the following features:
- Customer ID
- Age
- Gender
- Tenure (number of years the customer has been with the bank)
- Balance
- Number of products
- Credit score
- Whether the customer has a credit card
- Whether the customer is an active member
- Estimated salary
- Churn status (target variable)

## Model Architecture

The ANN model architecture consists of multiple layers of interconnected neurons, including input, hidden, and output layers. We have utilized popular libraries like TensorFlow or PyTorch to implement and train the neural network. The model takes the customer data as input and produces a binary output indicating whether the customer is likely to churn or not.

## Evaluation

The performance of the model can be evaluated using various metrics such as accuracy, precision, recall, and F1-score. Additionally, visualizations such as confusion matrices and ROC curves can provide insights into the model's performance across different thresholds.
