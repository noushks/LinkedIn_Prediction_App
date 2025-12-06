#!/usr/bin/env python
# coding: utf-8

# # Anoushka Mukherjee
# ## Final Project
# ### December 9th, 2025
# 
# ---

# In[2]:


#Importing all required datasets:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import streamlit as st


# ---

# #### Q1: Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

# In[16]:


s = pd.read_excel("social_media_usage.xlsx")


# In[17]:


#checking the dimensions of the data frame: 

s_dimensions = s.shape
print(s_dimensions)


# ---

# #### Q2: Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[18]:


def clean_sm(x):
    x = np.where(x == 1,1,0)
    return x
#creating toy dataframe:
s_toy = pd.DataFrame({
    "web1h" : [1,0,3],
    "gender" : [1,4,7]
})
#testing: 
s_toy["web1h_clean"] = clean_sm(s_toy["web1h"])
s_toy["gender_clean"] = clean_sm(s_toy["gender"])

print(s_toy)


# ---

# #### Q3: Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[20]:


ss = pd.DataFrame({
    "sm_li" : clean_sm(s["web1h"]),
    "income" : s["income"].where(s["income"] <= 9),
    "education" : s["educ2"].where(s["educ2"] <= 8),
    "parent"  : clean_sm(s["par"]), 
    "married" : np.where(s["marital"] == 1, 1, 0), 
    "female"  : np.where(s["gender"] == 2, 1, 0),
    "age"     : s["age"].where(s["age"] <= 98)
})
ss = ss.dropna()
print(ss.head())
print("Shape:", ss.shape)


# ---

# #### Q4: Create a target vector (y) and feature set (X)

# In[21]:


#Taking target vector y = LinkedIn user, and feature set x = income level, education level, parent, married, female and age: 

y = ss["sm_li"]

x = ss[["income", "education", "parent", "married", "female", "age"]]

print ("y.shape = ", y.shape)
print ("x.shape = ", x.shape)


print("\nTarget sample:")
print(y.head())

print("\nFeature sample:")
print(x.head())


# ---

# #### Q5: Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# In[22]:


#splitting the data into an 80/20 train-test split: 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size = 0.20,
    random_state = 15,
    stratify = y            
)

print("x_train Shape:", x_train.shape)
print("y_train Shape:", y_train.shape)
print("x_test Shape:", x_test.shape)
print("y_test Shape:", y_test.shape)


# #### Here, we created 4 new objects: x_train, y_train, x_test, y_test. x_train is used to train the model. An 80% train split means that 80% of the feature data comes from this dataset. x_test is used to evalute (on unseen data), how well the model has been trained. y_train is the target label (LinkedIn users) used to evaluate the predictions against the feature set to help reduce errors while training. y_test is an object created that is used to measure model performance. 

# ---

# #### Q6: Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# In[23]:


#loaded LogisticRegression from sklearn.linear_model

logreg = LogisticRegression(class_weight = "balanced", max_iter = 1000)

#fitting the model to the training data
logreg.fit(x_train, y_train)

# print("Model Training Complete")


# ---

# #### Q7: Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# In[24]:


y_pred = logreg.predict(x_test)

#to check model accuracy: 
accuracy = accuracy_score(y_test, y_pred)
print ("Model Accuracy:", round(accuracy, 4))

#confusion matrix: 
conf_mat = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix: ")
print(conf_mat)


# #### Confusion matrix is a 2 x 2 table that is used to evaluate the performance of a classification model by comparing the predictions against the actual values. It helps us get a detailed breakdown of correct and incorrect predictions, which can be eventually used to calculate other performance metrics like accuracy, precision, recall, and F1 scores. 
# #### It shows us the true positive (108), true negative (63), false positive (60), false negative (21). 
# 
# #### Our model has an accuracy of 0.6786, which means that the model predicts LinkedIn usage status correctly ~68% of the time (with regard to the test dataset). 
# 
# #### Top left quadrant: True negative (TN) indicates that the model correctly identified 108 people who do not use LinkedIn. 
# #### Top right quadrant: False positive (FP) indicates that the model predicted LinkedIn usage for 60 people, but they do not use LinkedIn
# #### Bottom left quadrant: False negative (FN) indicates that the model predicted no LinkedIn use for 21 people, but they actually do use LinkedIn
# #### Bottom right quadrant: True positive (TP) indicates that the model correctly predicted 63 people who use LinkedIn
# 

# ---

# #### Q8: Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

# In[25]:


conf_mat_df = pd.DataFrame(conf_mat,
                     index=["Actual_Negative (0)", "Actual_Positive (1)"],
                     columns=["Predicted_Negative (0)", "Predicted_Positive (1)"])
print(conf_mat_df)


# ---

# #### Q9: Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# #### Calculating using formula: 
# #### **Precision** = TP/(TP+FP) = 63/(63+60) = **0.5121** = **0.51**
# #### **Recall** = TP/(TP+FN) = 63/(63+21) = **0.75**
# #### **F1 Score** = 2 * ((Precision * Recall)/ (Precision + Recall)) = 2*((0.5121 * 0.75)/(0.5121 + 0.75)) = 2*(0.384075/1.2621) =2 * 0.3043 = **0.61**
# 

# In[26]:


print(classification_report(y_test, y_pred))


# ---

# #### Q10: Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

# In[27]:


#Let's assume person A is 42 years old: 
person_A = np.array([[8, 7, 0, 1, 1, 42]])

#Let's assume person B is 82 years old: 
person_B = np.array([[8, 7, 0, 1, 1, 82]])

# Predicting probabilities (column 1 = probability of LinkedIn use = class "1")
prob_A = logreg.predict_proba(person_A)[0][1]
prob_B = logreg.predict_proba(person_B)[0][1]

print(f"Probability Person A (age 42) uses LinkedIn: {prob_A:.4f}")
print(f"Probability Person B (age 82) uses LinkedIn: {prob_B:.4f}")


# In[ ]:





# In[ ]:





# In[ ]:




