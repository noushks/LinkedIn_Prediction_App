#!/usr/bin/env python
# coding: utf-8

# In[ ]:


st.title("Predicting LinkedIn Usage")
st.markdown(
    """
This app uses a logistic regression model (from your final project)  
to predict whether someone uses LinkedIn based on their **income, education,
parental status, marital status, gender, and age**.
"""
)


# In[ ]:


# Load data + model
ss, x, y = load_and_prepare_data()
model, accuracy, conf_mat, report = train_model(x, y)


# In[ ]:


# Sidebar:

st.sidebar.header("Enter Person's Information")

income = st.sidebar.slider("Income category (1 = lowest, 9 = highest)", 1, 9, 8)
education = st.sidebar.slider("Education level (1–8)", 1, 8, 7)

parent_label = st.sidebar.radio("Is the person a parent?", ["No", "Yes"])
parent = 1 if parent_label == "Yes" else 0

married_label = st.sidebar.radio("Marital status", ["Not married", "Married"])
married = 1 if married_label == "Married" else 0

gender_label = st.sidebar.radio("Gender", ["Male", "Female"])
female = 1 if gender_label == "Female" else 0

age = st.sidebar.slider("Age", 18, 98, 42)


# In[ ]:


# Turn inputs into model-ready array
person = np.array([[income, education, parent, married, female, age]])


# In[ ]:


# --- Main: model performance ---

st.subheader("Model Performance on Test Data")
st.write(f"**Accuracy:** {accuracy:.3f}")

conf_df = pd.DataFrame(
    conf_mat,
    index=["Actual: No LinkedIn (0)", "Actual: LinkedIn (1)"],
    columns=["Predicted: No LinkedIn (0)", "Predicted: LinkedIn (1)"],
)
st.write("**Confusion Matrix:**")
st.dataframe(conf_df)

st.write("**Classification Report (precision, recall, F1):**")
report_df = pd.DataFrame(report).T
st.dataframe(report_df.style.format(precision=3))


# In[ ]:


# --- Prediction for entered person ---

st.subheader("Prediction for the Entered Person")

if st.button("Predict LinkedIn Usage"):
    prob = model.predict_proba(person)[0][1]  # probability of class 1 = uses LinkedIn
    pred_class = model.predict(person)[0]

    st.metric("Predicted probability of LinkedIn use", f"{prob:.2%}")

    if pred_class == 1:
        st.success("Model prediction: **This person uses LinkedIn (class 1).**")
    else:
        st.warning("Model prediction: **This person does NOT use LinkedIn (class 0).**")

    st.caption(
        "Note: This uses the logistic regression model from your Programming 2 final project."
    )


# In[ ]:


# Optional: show a preview of the cleaned data
with st.expander("Show cleaned dataset (ss)"):
    st.write(ss.head())
    st.write(f"Shape: {ss.shape}")

