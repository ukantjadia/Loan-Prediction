# Project Name

Loan Prediction

## Table of Contents

- [Project Name](#project-name)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Data](#data)
  - [Models](#models)
  - [Usage](#usage)
  - [Conclusion](#conclusion)

## Introduction

This project aims to create a machine learning model that can predict whether a loan application should be approved or denied based on various factors such as credit history, income, loan amount, etc.

## Data

The dataset used for this project was sourced from Kaggle. It consists of loan applications with 12 features such as Loan ID, Gender, Marital Status, Education, etc.

## Models

After experimenting with various models, I selected the Logistic Regression model as our final model for predicting loan approvals. The model was chosen based on its high accuracy on the test data set, as well as its ability to handle the categorical data present in the loan dataset.

The logistic regression model provided an accuracy of 90% on the test dataset, which was higher than the other models tested such as Decision Tree Classifier, K-Neighbours, and SVC.

We believe that this model will perform well on new data sets and can be used to predict loan approvals accurately

## Usage

To use this model, clone the repository and run the loan_approval.py file. The user will be prompted to enter the required loan application details such as credit history, income, loan amount, etc. The model will then predict whether the loan application should be approved or denied based on the entered details.

## Conclusion

This project demonstrates the use of machine learning algorithms to build a loan approval model that can help banks and other financial institutions to make more informed lending decisions.