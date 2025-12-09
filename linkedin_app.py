import streamlit as st
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load and prepare the data
s = pd.read_excel("social_media_usage.xlsx")

# Creating a function: 
def clean_sm(x):
    """Return 1 if value is 1, else 0."""
    return np.where(x == 1, 1, 0)

#Creating a new variable ss: 
ss = pd.DataFrame({
    "sm_li": clean_sm(s["web1h"]),                         
    "income": s["income"].where(s["income"] <= 9),         
    "education": s["educ2"].where(s["educ2"] <= 8),        
    "parent": clean_sm(s["par"]),                          
    "married": np.where(s["marital"] == 1, 1, 0),          
    "female": np.where(s["gender"] == 2, 1, 0),            # female = 1
    "age": s["age"].where(s["age"] <= 98)                
})

# Drop rows with missing values
ss = ss.dropna()

# Features and target variables:
feature_cols = ["income", "education", "parent", "married", "female", "age"]
x = ss[feature_cols]
y = ss["sm_li"]

# Train/test split 80/20:
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.20,
    random_state=15,
    stratify=y
)

# Training the logistic regression model:
logreg = LogisticRegression(class_weight="balanced", max_iter=1000)
logreg.fit(x_train, y_train)

# Evaluating on test set:
y_pred = logreg.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


# 2. Streamlit app layout

st.title("LinkedIn Usage Prediction App")
st.write(
    "Do you statistically use LinkedIn?" 
    " This app uses a logistic regression model to predict whether a person is likely to use LinkedIn, based on demographic information."
)

st.markdown("---")

# 3. Input sidebar

st.sidebar.header("Tell me a bit about yourself")

income_options = [
    "1. Less than $10,000",
    "2. $10k–$20k",
    "3. $20k–$30k",
    "4. $30k–$40k",
    "5. $40k–$50k",
    "6. $50k–$75k",
    "7. $75k–$100k",
    "8. $100k–$150k",
    "9. $150k+"
]

education_options = [
    "1. Less than high school",
    "2. High school incomplete",
    "3. High school graduate/GED",
    "4. Some college, no degree",
    "5. Associate degree",
    "6. Bachelor’s degree",
    "7. Some postgraduate, no degree",
    "8. Postgraduate/professional degree"
]

income_label = st.sidebar.selectbox("Income level", income_options)
income = income_options.index(income_label) + 1   

education_label = st.sidebar.selectbox("Education level", education_options)
education = education_options.index(education_label) + 1   


parent_str = st.sidebar.radio("Is this person a parent?", ("No", "Yes"))
parent = 1 if parent_str == "Yes" else 0

marital_options = [
    "1. Married",
    "2. Living with a partner",
    "3. Divorced",
    "4. Separated",
    "5. Widowed",
    "6. Never been married",
    "8. Don't know",
    "9. Don't want to disclose"
]

marital_choice = st.sidebar.selectbox("Marital Status", marital_options)

# Converting martial choice to numeric
marital_code = int(marital_choice.split(".")[0])

# Converting marital choice to binary:
# 1 = married, 0 = all others
married = 1 if marital_code == 1 else 0

female_str = st.sidebar.radio("Gender", ("Male", "Female"))
female = 1 if female_str == "Female" else 0

age = st.sidebar.slider("Age", 18, 98, 35)


# 4. Making a prediction:

st.subheader("Prediction")

if st.button("Predict LinkedIn Usage"):
    
    person_features = np.array([[income, education, parent, married, female, age]])

    # Predicted class and probability
    pred_class = logreg.predict(person_features)[0]
    prob_linkedin = logreg.predict_proba(person_features)[0][1]  # probability of class 1

    if pred_class == 1:
        st.success("Predicted class: **LinkedIn user**")
    else:
        st.markdown(
            """
            <div style="
                background-color: rgba(255, 0, 0, 0.25);
                padding:12px;
                border-radius:6px;
                color:white;
                font-weight:bold;
                font-size:18px;">
                Predicted class: <b>Not a LinkedIn user</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write(f"**Probability this person uses LinkedIn:** {prob_linkedin:.1%}")

    #Creating usage rate and size visualization: 

ss = ss.copy()
# Age groups
age_bins = [0, 29, 44, 59, 120]
age_labels = ["18–29", "30–44", "45–59", "60+"]
ss["age_group"] = pd.cut(ss["age"], bins=age_bins, labels=age_labels, right=True, include_lowest=True)

# Income bands (adjust ranges if you prefer)
inc_bins = [0, 3, 6, 9]
inc_labels = ["Low (1–3)", "Mid (4–6)", "High (7–9)"]
ss["income_band"] = pd.cut(ss["income"], bins=inc_bins, labels=inc_labels, right=True, include_lowest=True)

# Gender label
ss["gender_label"] = ss["female"].map({1: "Female", 0: "Male"})
