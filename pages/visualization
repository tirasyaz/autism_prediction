import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set up the visual style
sns.set(style="whitegrid")

# Load the dataset
file_path = '/mnt/data/Toddler Autism dataset July 2018.csv'
data = pd.read_csv(file_path)

# Title and description
st.title("Toddler Autism Dataset Visualization")
st.write("""
This app visualizes various parameters of the Toddler Autism dataset, including distributions of age, gender counts, screening test outcomes, and feature correlations.
""")

# Display the dataset
if st.checkbox("Show Dataset"):
    st.write(data)

# Sidebar for selecting visualizations
st.sidebar.header("Select Visualization")
visualization = st.sidebar.selectbox("Choose a visualization:", 
                                     ["Age Distribution", "Gender Count", "Screening Test Outcome", "Correlation Heatmap"])

# 1. Age Distribution
if visualization == "Age Distribution":
    st.subheader("Age Distribution of Toddlers")
    fig, ax = plt.subplots()
    sns.histplot(data['Age'], bins=10, kde=True, ax=ax)
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# 2. Gender Count
elif visualization == "Gender Count":
    st.subheader("Count of Toddlers by Gender")
    fig, ax = plt.subplots()
    sns.countplot(x='Gender', data=data, palette="viridis", ax=ax)
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# 3. Screening Test Outcome
elif visualization == "Screening Test Outcome":
    st.subheader("Screening Test Outcomes")
    fig, ax = plt.subplots()
    sns.countplot(x='Screening_Test', data=data, palette="pastel", ax=ax)
    ax.set_xlabel("Screening Outcome")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# 4. Correlation Heatmap
elif visualization == "Correlation Heatmap":
    st.subheader("Correlation Heatmap of Numerical Features")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", square=True, ax=ax)
    st.pyplot(fig)

# Additional options
st.sidebar.header("Additional Options")
if st.sidebar.checkbox("Show Summary Statistics"):
    st.subheader("Summary Statistics")
    st.write(data.describe())
