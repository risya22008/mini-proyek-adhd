import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="ADHD Prediction App", layout="wide")

def load_model_and_scaler():
    # Generate sample dataset (replace this with your actual dataset)
    np.random.seed(42)
    n_samples = 1000
    
    X = np.random.rand(n_samples, 4)  # 4 features: bdi, audit, aas, asrs
    y = (X.sum(axis=1) > 2).astype(int)  # Binary classification
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def main():
    st.title("ADHD Prediction System")
    st.write("Please fill out the following questionnaires to assess ADHD likelihood.")
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Create tabs for different questionnaires
    tab1, tab2, tab3, tab4 = st.tabs(["BDI", "AUDIT", "AAS", "ASRS"])
    
    with tab1:
        st.header("Beck Depression Inventory (BDI)")
        bdi_questions = [
            "Sadness", "Pessimism", "Past Failure", "Loss of Pleasure", "Guilt",
            "Punishment Feelings", "Self-Dislike", "Self-Criticalness", "Suicidal Thoughts",
            "Crying", "Agitation", "Loss of Interest", "Indecisiveness", "Worthlessness",
            "Energy Loss", "Sleep Changes", "Irritability", "Appetite Changes",
            "Concentration Difficulty", "Fatigue"
        ]
        bdi_scores = []
        for q in bdi_questions:
            score = st.select_slider(
                f"{q} (0-3):",
                options=[0, 1, 2, 3],
                key=f"bdi_{q}"
            )
            bdi_scores.append(score)
        bdi_total = sum(bdi_scores)
        st.write(f"BDI Total Score: {bdi_total}")

    with tab2:
        st.header("Alcohol Use Disorders Identification Test (AUDIT)")
        audit_questions = [
            "Frequency of Drinking", "Typical Quantity Consumed",
            "Frequency of Heavy Drinking Episodes", "Impaired Control Over Drinking",
            "Increased Salience of Drinking", "Impaired Daily Functioning Due to Drinking",
            "Guilt About Drinking", "Injury Due to Drinking",
            "Concerns From Others About Drinking", "Alcohol Dependency Indicators"
        ]
        audit_scores = []
        for q in audit_questions:
            score = st.select_slider(
                f"{q} (0-4):",
                options=[0, 1, 2, 3, 4],
                key=f"audit_{q}"
            )
            audit_scores.append(score)
        audit_total = sum(audit_scores)
        st.write(f"AUDIT Total Score: {audit_total}")

    with tab3:
        st.header("Adult ADHD Self-Report Scale (ASRS)")
        asrs_questions = [
            "Kesulitan memperhatikan detail", "Kesulitan mempertahankan perhatian",
            "Kesulitan mendengarkan", "Tidak menyelesaikan tugas", "Kesulitan mengatur tugas",
            "Menghindari tugas mental", "Sering kehilangan barang", "Mudah terganggu",
            "Sering lupa aktivitas", "Gelisah", "Kesulitan duduk diam", 
            "Merasa gelisah fisik", "Selalu sibuk", "Berbicara berlebihan",
            "Membuat jawaban sebelum pertanyaan selesai", "Kesulitan menunggu giliran",
            "Menyela orang lain", "Tidak dapat menahan keinginan"
        ]
        asrs_scores = []
        for q in asrs_questions:
            score = st.select_slider(
                f"{q} (0-4):",
                options=[0, 1, 2, 3, 4],
                key=f"asrs_{q}"
            )
            asrs_scores.append(score)
        asrs_total = sum(asrs_scores)
        st.write(f"ASRS Total Score: {asrs_total}")

    with tab4:
        st.header("Anxiety Assessment Scale (AAS)")
        aas_questions = [
            "Merasakan mati rasa", "Merasa panas tiba-tiba", "Merasa lemah",
            "Rasa gemetar", "Merasa takut", "Kesulitan bernapas", "Merasa tercekik",
            "Jantung berdebar", "Merasa gugup", "Merasa takut kehilangan kendali",
            "Merasa gemetar di bagian tubuh", "Masalah pencernaan", "Merasa pusing",
            "Merasa tegang", "Sulit untuk santai", "Takut tubuh gagal berfungsi",
            "Wajah memerah", "Merasa berkeringat", "Merasa mudah terkejut",
            "Merasa kesulitan fokus", "Sulit tidur karena cemas"
        ]
        aas_scores = []
        for q in aas_questions:
            score = st.select_slider(
                f"{q} (0-3):",
                options=[0, 1, 2, 3],
                key=f"aas_{q}"
            )
            aas_scores.append(score)
        aas_total = sum(aas_scores)
        st.write(f"AAS Total Score: {aas_total}")

    # Prediction section
    st.header("Prediction")
    if st.button("Calculate ADHD Prediction"):
        # Prepare input data
        input_data = np.array([[bdi_total, audit_total, aas_total, asrs_total]])
        
        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Display results
        st.subheader("Results")
        st.write(f"ADHD Likelihood: {probability:.2%}")
        
        if prediction == 1:
            st.warning("Based on the responses, there might be indicators of ADHD. Please consult with a healthcare professional for a proper diagnosis.")
        else:
            st.success("Based on the responses, ADHD indicators are below the threshold. However, if you have concerns, please consult with a healthcare professional.")
        
        # Display summary of scores
        st.subheader("Summary of Scores")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("BDI Score", bdi_total)
        with col2:
            st.metric("AUDIT Score", audit_total)
        with col3:
            st.metric("ASRS Score", asrs_total)
        with col4:
            st.metric("AAS Score", aas_total)

if __name__ == "__main__":
    main()