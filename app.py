import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import altair as alt
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Caching Data Loading ---
@st.cache_data
def load_data(file_path):
    """Loads the student performance dataset from the provided CSV file."""
    try:
        # The new CSV uses a comma separator
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it is in the same directory as the app.py script.")
        return None
    except Exception as e:
        st.error(f"An error occurred while reading the CSV file: {e}")
        return None

# --- Model Training ---
@st.cache_resource
def train_model(data):
    """Trains a RandomForestClassifier model on the new dataset."""
    # Define features (all columns except student_name)
    features = [col for col in data.columns if col != 'student_name']
    
    # Create the target variable: 'pass_status'
    # Calculate the average score and determine pass/fail (threshold >= 40)
    data['average_score'] = data[features].mean(axis=1)
    data['pass_status'] = (data['average_score'] >= 40).astype(int)
    
    target = 'pass_status'
    
    # Drop non-feature columns for training
    X = data[features]
    y = data[target]
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, features, accuracy, data # Return modified data for visualization

# --- Main Application ---
def main():
    """The main function that runs the Streamlit application."""
    
    # --- Sidebar ---
    with st.sidebar:
        st.title("🎓 Student Performance Predictor")
        st.markdown("This application predicts a student's overall pass/fail status based on their subject marks.")
        
        st.header("Input Student Marks")
        st.markdown("Adjust the sliders to input the student's marks for each subject.")

        # Create input sliders for each subject's marks
        maths_marks = st.slider("Maths Marks", 0, 100, 50)
        science_marks = st.slider("Science Marks", 0, 100, 50)
        english_marks = st.slider("English Marks", 0, 100, 50)
        social_studies_marks = st.slider("Social Studies Marks", 0, 100, 50)
        language_marks = st.slider("Language Marks", 0, 100, 50)

    # --- Main Panel ---
    st.title("Prediction and Analysis")
    st.markdown("---")

    # Load data and train model
    df = load_data('student-mat.csv')
    
    if df is not None:
        model, features, accuracy, df_processed = train_model(df.copy())

        # Prediction Logic
        input_data = pd.DataFrame([[
            maths_marks, science_marks, english_marks, social_studies_marks, language_marks
        ]], columns=features)
        
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Display Prediction
        st.subheader("Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.success("The model predicts the student will **PASS**.")
            else:
                st.error("The model predicts the student will **FAIL**.")
        with col2:
            st.metric(label="Model Confidence", value=f"{prediction_proba[prediction]*100:.2f}%")
        
        st.info(f"The model used for this prediction has an accuracy of **{accuracy*100:.2f}%** on a held-out test set.")
        
        st.markdown("---")

        # Data Analysis and Visualization Section
        st.header("Exploratory Data Analysis")
        
        # Show raw data
        if st.checkbox("Show Processed Data (with average score and pass status)"):
            st.dataframe(df_processed)

        # Visualizations
        st.subheader("Data Visualizations")
        
        col_vis1, col_vis2 = st.columns(2)

        with col_vis1:
            st.markdown("#### Distribution of Average Scores")
            chart_avg = alt.Chart(df_processed).mark_bar().encode(
                alt.X("average_score", bin=alt.Bin(maxbins=20), title="Average Score"),
                alt.Y('count()', title="Number of Students"),
                tooltip=['count()']
            ).interactive()
            st.altair_chart(chart_avg, use_container_width=True)

        with col_vis2:
            st.markdown("#### Subject Marks vs. Pass/Fail")
            # Create a pass/fail column for visualization
            df_processed['Result'] = df_processed['pass_status'].apply(lambda x: 'Pass' if x == 1 else 'Fail')
            
            # Melt the dataframe to have subjects in one column and marks in another
            df_melted = df_processed.melt(
                id_vars=['Result'], 
                value_vars=['maths_marks', 'science_marks', 'english_marks', 'social_studies_marks', 'language_marks'],
                var_name='Subject',
                value_name='Marks'
            )

            chart_subjects = alt.Chart(df_melted).mark_boxplot().encode(
                x=alt.X('Subject:N', title="Subject"),
                y=alt.Y('Marks:Q', title="Marks"),
                color=alt.Color('Result:N', scale=alt.Scale(domain=['Pass', 'Fail'], range=['#2ca02c', '#d62728'])),
            ).properties(
                height=300
            ).interactive()
            st.altair_chart(chart_subjects, use_container_width=True)

if __name__ == "__main__":
    main()