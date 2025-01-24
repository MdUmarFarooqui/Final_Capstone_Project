# Sentiment Analysis: Capstone Project 2
## Problem Statement:
The dataset consists of over 34,000 consumer reviews for Amazon brand products, including items like Kindle, Fire TV Stick, and more. Key attributes in the dataset include product brand, category, primary category, review titles, review text, and sentiment. Sentiment is classified into three categories: Positive, Negative, and Neutral. The objective is to build a model capable of predicting the sentiment for unseen reviews, leveraging multiple features and textual data.

## Project Workflow and Approach:
### 1. Initial Exploration and Problem Identification:
* Perform Exploratory Data Analysis (EDA) to understand the dataset structure and distribution.
* Analyze examples of positive, negative, and neutral reviews to gain qualitative insights into sentiment patterns.
Examine the distribution of sentiment classes. 
Address the identified class imbalance problem for effective model training and evaluation.
### 2. Feature Engineering and Baseline Model:
Convert textual reviews into numerical representations using TF-IDF scores.
Train a Multinomial Naive Bayes classifier as a baseline model. Note: Due to class imbalance, the model is likely to overpredict the dominant class (positive sentiment).
### 3. Addressing Class Imbalance:
Implement techniques such as oversampling (e.g., SMOTE) or undersampling to balance the dataset.
Evaluate model performance using appropriate metrics for imbalanced data, including:
Precision, Recall, F1-Score, and AUC-ROC Curve.
Optimize models based on the F1-Score, as it balances precision and recall effectively.
### 4. Advanced Machine Learning Models:
Train tree-based classifiers such as Random Forest and XGBoost.
Leverage these algorithms’ inherent ability to handle imbalanced classes through hyperparameter tuning (e.g., class weights).
### 5. Enhanced Features and Ensembles:
Engineer a sentiment score from review text to create an additional feature. Incorporate this feature into models and evaluate its impact on performance.
Experiment with ensemble approaches such as combining XGBoost with oversampled Multinomial Naive Bayes.
### 6. Deep Learning Approach (LSTM and GRU):
Implement LSTM (Long Short-Term Memory) networks for sentiment classification, optimizing parameters such as:
Top words, embedding length, dropout rate, epochs, and number of layers.
Explore GRU (Gated Recurrent Units) as an alternative to LSTM.
Compare the performance of neural networks with traditional machine learning models.
Optimize deep learning models using techniques like Grid Search, Random Search, and Cross-Validation to identify the best hyperparameter settings.
### 7. Topic Modeling and Clustering:
Cluster reviews to identify common themes, such as:
Reviews discussing the product as a gift option.
Comments on aesthetic design or battery performance.
Perform Topic Modeling using techniques like:
Latent Dirichlet Allocation (LDA).
Non-Negative Matrix Factorization (NMF).
Name clusters meaningfully based on identified patterns and insights.
### Deliverables and Insights:
Develop a predictive model that classifies sentiments with high accuracy.
Provide a comparative analysis of model performance between traditional machine learning and neural networks.
Extract actionable insights from topic modeling to better understand consumer sentiment and preferences.
Summarize findings and present model outputs through visualizations and interpretability metrics.
