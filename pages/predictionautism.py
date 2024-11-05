import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/mnt/data/Toddler Autism dataset July 2018.csv'
data = pd.read_csv(file_path)

# Set up the Streamlit app
st.title("Toddler Autism Dataset Visualization")
st.write("This app visualizes the Toddler Autism dataset parameters.")

# Display the dataset
if st.checkbox("Show Raw Data"):
    st.write(data)

# Basic Information
st.subheader("Dataset Information")
st.write("Number of Rows:", data.shape[0])
st.write("Number of Columns:", data.shape[1])
st.write("Column Names:", data.columns.tolist())

# Select column for visualization
selected_column = st.selectbox("Select a column to visualize", data.columns)

# Display selected column distribution
st.subheader(f"Distribution of {selected_column}")

fig, ax = plt.subplots()
if data[selected_column].dtype == 'object':
    sns.countplot(y=data[selected_column], ax=ax, palette="viridis")
    st.pyplot(fig)
else:
    sns.histplot(data[selected_column], kde=True, ax=ax, color="blue")
    st.pyplot(fig)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
if len(numeric_columns) > 1:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data[numeric_columns].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.write("Not enough numeric columns for a correlation heatmap.")

# Pairplot for selected subset of columns
st.subheader("Pairplot for Selected Columns")
selected_columns = st.multiselect("Choose columns for pairplot", numeric_columns)
if len(selected_columns) > 1:
    fig = sns.pairplot(data[selected_columns])
    st.pyplot(fig)
else:
    st.write("Select at least two columns for a pairplot.")

# Run with: streamlit run app.py

