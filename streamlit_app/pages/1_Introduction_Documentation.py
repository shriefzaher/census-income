import streamlit as st

# Set page title
st.title("Introduction and Documentation")

# Add documentation content
st.markdown("""

## Overview

## Introduction for Your Streamlit App: Census Income Prediction and Data Analysis Project Overview:

> *Welcome* to the Census Income Prediction and Data Analysis App!

This project aims to provide insights into demographic and employment factors that influence income levels.

Using a dataset derived from the U.S. Census, we leverage machine learning models

to analyze and predict income levels. Our goal is to predict whether an individual's income exceeds $50,000 per year based on key attributes

such as age, education, occupation, workclass, marital status, and more..

  

## Data Exploration and Visualization:

  

The app offers interactive tools that let you visualize trends, distributions, and
correlations within the dataset. For example, you can see how education level affects income, the distribution of income across different work classes, or explore gender and income trends.

Key visualizations include bar charts, pie charts, box plots, and confusion matrices that show the model's performance. You can provide a deep dive into the data.

  

## Machine Learning Models:

  

This app utilizes a variety of machine learning algorithms, including Decision Trees, Logistic Regression, Random Forests, Naive Bayes, XGBoost, and an ensemble Voting Classifier to predict income.

The models are trained using demographic data (age, education, marital status, etc.) and employment-related data (occupation, workclass, hours worked), and predict whether a person earns more than $50,000 per year.

After training, the models are evaluated using test data, and the app displays metrics like accuracy, precision, recall, and F1-score, as well as confusion matrices to show the model's performance.

  
  

## How to Use This App

**1- Navigate Using the Sidebar:** 
            
- On the left-hand sidebar, you will find different sections of the app.
- Each page focuses on specific functionalities, such as data visualization or model inference.

  

**2- Model Inference:**

-   Select the **Model Inference** option from the sidebar.
-   Once you navigate to the **Model Inference** section, the pre-trained machine learning models (such as Decision Trees, Logistic Regression, XGBoost, etc.) will automatically run predictions on the dataset.
-   These models predict whether an individual’s income is **<=50K** or **>50K**, based on the various features provided in the dataset.

  


""")
            

