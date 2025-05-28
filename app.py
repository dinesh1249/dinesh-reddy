import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("car_evaluation 1.csv", names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

data = load_data()

st.title("Car Evaluation Classifier")
st.write("This app uses a Decision Tree to classify cars into categories based on various attributes.")

# Encode features
X = pd.get_dummies(data.iloc[:, :-1])
Y = data['class']

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)

# Accuracy
Y_pred = clf.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
st.write(f"Model Accuracy: **{acc:.2f}**")

# User input
st.subheader("Predict Car Class")

# Dynamic form
def user_input_features():
    buying = st.selectbox('Buying Price', data['buying'].unique())
    maint = st.selectbox('Maintenance Price', data['maint'].unique())
    doors = st.selectbox('Number of Doors', data['doors'].unique())
    persons = st.selectbox('Capacity (persons)', data['persons'].unique())
    lug_boot = st.selectbox('Luggage Boot Size', data['lug_boot'].unique())
    safety = st.selectbox('Safety', data['safety'].unique())

    user_df = pd.DataFrame([[buying, maint, doors, persons, lug_boot, safety]],
                           columns=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
    return user_df

input_df = user_input_features()

# Encode input like training data
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

# Prediction
prediction = clf.predict(input_encoded)
st.write(f"### Predicted Car Class: `{prediction[0]}`")
