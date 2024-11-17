import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Function to load and preprocess data
def load_and_preprocess_data(uploaded_file):
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Drop unnecessary columns
    df.drop(['comments', 'state', 'Timestamp'], axis=1, inplace=True)
    
    # Clean gender
    male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr", "cis man", "Cis Male", "cis male"]
    trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
    female_str = ["cis female", "f", "female", "woman", "femake", "female ", "cis-female/femme", "female (cis)", "femail"]
    
    df['Gender'] = df['Gender'].str.lower().apply(
        lambda x: 'male' if x in male_str else
                  'female' if x in female_str else
                  'trans' if x in trans_str else 'other'
    )
    
    # Handle missing values
    df['self_employed'] = df['self_employed'].fillna('No')
    df['work_interfere'] = df['work_interfere'].fillna("Don't know")
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Age processing
    df.loc[df['Age'] < 18, 'Age'] = df['Age'].median()
    df.loc[df['Age'] > 120, 'Age'] = df['Age'].median()
    
    # Create age ranges
    df['age_range'] = pd.cut(df['Age'], [0, 20, 30, 65, 100], labels=["0-20", "21-30", "31-65", "66-100"])
    
    # Drop country column
    df = df.drop(['Country'], axis=1)
    
    return df

# Function to encode features
def encode_features(df):
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object' or df[column].dtype == 'category':
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])
    
    # Scale Age
    scaler = MinMaxScaler()
    df['Age'] = scaler.fit_transform(df[['Age']])
    
    return df, label_encoders, scaler

# Function to create model
def create_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(max_depth=None, 
                                    min_samples_leaf=8, 
                                    min_samples_split=2, 
                                    n_estimators=20, 
                                    random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy, y_pred

# Main Streamlit App
def main():
    st.title('Mental Health in Tech Survey Analysis')
    st.write("""
    This application analyzes mental health survey data from tech companies.
    Upload your survey data to begin the analysis.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load and preprocess data
        df = load_and_preprocess_data(uploaded_file)
        
        # Show data overview
        st.header('Data Overview')
        st.write('Sample of the preprocessed data:')
        st.dataframe(df.head())
        
        # Data Analysis Section
        st.header('Data Analysis')
        
        # Gender Distribution
        st.subheader('Gender Distribution')
        gender_counts = df['Gender'].value_counts()
        fig = px.pie(values=gender_counts.values, names=gender_counts.index)
        st.plotly_chart(fig)
        
        # Age Distribution
        st.subheader('Age Distribution')
        fig = px.histogram(df, x='Age', nbins=20)
        st.plotly_chart(fig)
        
        # Treatment seeking by gender
        st.subheader('Treatment Seeking by Gender')
        treatment_by_gender = pd.crosstab(df['Gender'], df['treatment'])
        fig = px.bar(treatment_by_gender, barmode='group')
        st.plotly_chart(fig)
        
        # Model Training Section
        st.header('Model Training')
        
        # Encode features
        encoded_df, label_encoders, scaler = encode_features(df.copy())
        
        # Select features
        feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 
                        'care_options', 'anonymity', 'leave', 'work_interfere']
        X = encoded_df[feature_cols]
        y = encoded_df['treatment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
        
        # Train model
        model, accuracy, y_pred = create_model(X_train, X_test, y_train, y_test)
        
        # Show results
        st.subheader('Model Performance')
        st.write(f'Model Accuracy: {accuracy:.2%}')
        
        # Confusion Matrix
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm,
                        labels=dict(x="Predicted", y="Actual"),
                        x=['No Treatment', 'Treatment'],
                        y=['No Treatment', 'Treatment'],
                        text_auto=True)
        st.plotly_chart(fig)
        
        # Feature Importance
        st.subheader('Feature Importance')
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        fig = px.bar(importance_df, x='feature', y='importance')
        st.plotly_chart(fig)
        
        # Prediction Interface
        st.header('Make Predictions')
        st.write('Use this interface to make predictions for new data:')
        
        # Create input fields for each feature
        new_data = {}
        for feature in feature_cols:
            if feature == 'Age':
                new_data[feature] = st.slider('Age', 18, 100, 30)
            else:
                unique_values = sorted(df[feature].unique())
                new_data[feature] = st.selectbox(f'Select {feature}', unique_values)
        
        if st.button('Predict'):
            # Prepare input data
            input_df = pd.DataFrame([new_data])
            
            # Encode input data
            for column in input_df.columns:
                if column in label_encoders:
                    input_df[column] = label_encoders[column].transform([input_df[column][0]])[0]
                elif column == 'Age':
                    input_df[column] = scaler.transform([[input_df[column][0]]])[0, 0]
            
            # Make prediction
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)
            
            st.write('Prediction:', 'Treatment Recommended' if prediction[0] == 1 else 'No Treatment Recommended')
            st.write(f'Probability of needing treatment: {probability[0][1]:.2%}')

if __name__ == '__main__':
    main()
