import streamlit as st
import numpy as np
import pandas as pd
import os
from collections import Counter

st.set_page_config(page_title="ADHD Prediction App", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #89A8B2; 
    }
    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: #B3C8CF; 
    }
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 2.5rem;
        white-space: pre-wrap;
        background-color: #507687; 
        border-radius: 16px;
        padding: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #507687;
    }
    .scale-explanation {
        background-color: #89A8B2;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
     /* Styling untuk selectbox container */
    div[data-baseweb="select"] {
        background-color: #0B192C !important;
    }
    
    /* Styling untuk selectbox button */
    div[data-baseweb="select"] > div {
        background-color: #0B192C !important;
        color: white !important;
    }
    
    /* Styling untuk dropdown menu (listbox) */
    div[role="listbox"] {
    background-color: #0B192C !important;  
    border: 1px solid #0B192C;            
    color: white !important;             
}

    /* Styling untuk setiap option dalam dropdown */
    div[role="option"] {
    background-color: #0B192C !important; 
    color: white !important;             
    padding: 10px;
    border-radius: 4px;                 
}

    /* Styling untuk option yang di-hover */
    div[role="option"]:hover {
    background-color: #0B192C !important;  
    color: black !important;
}

    /* Styling untuk selected option */
    div[role="option"][aria-selected="true"] {
    background-color: #0B192C !important; 
    color: white !important;
}

    /* Mengubah warna button Calculate ADHD Prediction */
    .stButton button {
        background-color: #0B192C;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background-color: #ECDFCC;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)


# Custom ML Implementations
class CustomKNN:
    def _init_(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)

    def predict_proba(self, X):
        probabilities = []
        for x in X:
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            count_1 = sum(1 for label in k_nearest_labels if label == 1)
            prob_1 = count_1 / self.k
            probabilities.append([1 - prob_1, prob_1])
        return np.array(probabilities)

class CustomGaussianNB:
    def _init_(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def _calculate_likelihood(self, x, mean, var):
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return np.prod(1 / np.sqrt(2 * np.pi * var) * exponent)

    def predict_proba(self, X):
        probabilities = []
        for x in X:
            likelihoods = []
            for idx in range(len(self.classes)):
                likelihood = self._calculate_likelihood(x, self.mean[idx], self.var[idx])
                posterior = likelihood * self.priors[idx]
                likelihoods.append(posterior)

            total = sum(likelihoods)
            if total == 0:
                probabilities.append([1 / len(self.classes)] * len(self.classes))
            else:
                probabilities.append([p / total for p in likelihoods])

        return np.array(probabilities)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

class CustomScaler:
    def _init_(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / (self.std + 1e-8)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def load_model_and_scaler(model_type="naive_bayes", file_path="./ADHD.xlsx"):
    base_dir = os.path.dirname(__file__)
    full_file_path = os.path.join(base_dir, file_path)

    if not os.path.exists(full_file_path):
        raise FileNotFoundError(f"Dataset '{file_path}' tidak ditemukan.")

    data = pd.read_excel(full_file_path)

    feature_columns = [
        'bdi1_total', 'audit1_total', 'aas1_total', 'asrs1_total.x', 'bai1_total'
    ]
    target_column = 'adhd_label'

    X = data[feature_columns].values
    y = data[target_column].values

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    scaler = CustomScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "naive_bayes":
        model = CustomGaussianNB()
    else:
        model = CustomKNN(k=5)

    model.fit(X_train_scaled, y_train)

    # Calculate F1 Score manually on test data
    y_pred = model.predict(X_test_scaled)

    # Manually calculate F1 Score
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    tn = np.sum((y_pred == 0) & (y_test == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"F1 Score (Test Data): {f1:.2f}")

    return model, scaler

def save_input_to_csv(data, file_path):
    if not os.path.exists(file_path):
        data.to_csv(file_path, index=False)
    else:
        existing_data = pd.read_csv(file_path)
        updated_data = pd.concat([existing_data, data], ignore_index=True)
        updated_data.to_csv(file_path, index=False)

def get_treatment_recommendation(prediction, probability):
    if prediction == 1:
        if probability >= 0.8:
            return "High likelihood of ADHD. It is recommended to consult with a healthcare professional for a comprehensive evaluation and consider treatments such as cognitive behavioral therapy (CBT), medication, or lifestyle adjustments (e.g., improved time management and task structuring)."
        elif 0.5 <= probability < 0.8:
            return "Moderate likelihood of ADHD. It is recommended to seek a professional assessment for a more detailed diagnosis. Behavioral interventions or lifestyle changes may be beneficial."
        else:
            return "There is a possibility of ADHD. Consider monitoring symptoms and consult a healthcare professional if symptoms persist or worsen."
    else:
        return "ADHD is unlikely based on the assessment. However, if you experience any challenges related to focus, organization, or anxiety, consider discussing with a healthcare professional for further advice."

def main():

    DATA_FILE = "./adhd_data.csv"
    
    st.title("AttentionAid - ADHD Prediction System")
    image_path = r"./adhd.png"
    
    try:
        st.image(image_path, width=150)
    except Exception as e:
        st.error(f"Error loading image: {e}")
    st.write("Please fill out the following questionnaires to assess ADHD likelihood.")
    
    # Model selection
    model_type = st.sidebar.selectbox("Select Prediction Model", ["Naive Bayes", "K-Nearest Neighbors"])
    model_type_key = "naive_bayes" if model_type == "Naive Bayes" else "knn"
    
    model, scaler = load_model_and_scaler(model_type_key)
    
    # Create tabs for different questionnaires
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 BDI", "🍷 AUDIT", "🎯 ASRS", "😰 AAS", "😟BAI"])

    # BDI Tab
    with tab1:
        st.header("Beck Depression Inventory (BDI)")
        st.write("0: tidak ada")
        st.write("1: ringan")
        st.write("2: sedang")
        st.write("3: berat")
        bdi_questions = [
            "Kesedihan", "Pesimisme", "Kegagalan Masa Lalu", "Kehilangan Kesenangan", "Rasa Bersalah", "Perasaan Hukuman", "Tidak Suka Diri Sendiri", "Sikap Mengkritik Diri Sendiri", "Pikiran Bunuh Diri", "Menangis", "Agitasi", "Kehilangan Minat", "Keragu-raguan", "Tidak Berharga", "Kehilangan Energi", "Perubahan Tidur", "Mudah Tersinggung", "Perubahan Nafsu Makan", "Kesulitan Konsentrasi", "Kelelahan", "Berdiam Diri"
        ]
        bdi_scores = [st.select_slider(f"{q} (0-3):", options=[0, 1, 2, 3], key=f"bdi_{q}") 
                     for q in bdi_questions]
        bdi_total = sum(bdi_scores)
        st.write(f"BDI Total Score: {bdi_total}")

    # AUDIT Tab
    with tab2:
        st.header("Alcohol Use Disorders Identification Test (AUDIT)")
        st.write("0: tidak pernah")
        st.write("1: kadang kadang")
        st.write("2: sering")
        st.write("3: sangat sering")
        st.write("4: sangat tinggi frekuensinya")
        audit_questions = [
            "Frekuensi Minum", "Jumlah Konsumsi yang Biasa", "Kontrol yang Terganggu terhadap Minum",
            "Peningkatan Kepentingan Minum", "Fungsi Harian yang Terganggu Akibat Minum",
            "Rasa Bersalah Karena Minum", "Cedera Akibat Minum",
            "Kekhawatiran Orang Lain Mengenai Minum", "Indikator Ketergantungan Alkohol"
        ]
        audit_scores = [st.select_slider(f"{q} (0-4):", options=[0, 1, 2, 3, 4], key=f"audit_{q}") 
                       for q in audit_questions]
        audit_total = sum(audit_scores)
        st.write(f"AUDIT Total Score: {audit_total}")

    # ASRS Tab
    with tab3:
        st.header("Adult ADHD Self-Report Scale (ASRS)")
        st.write("0: tidak pernah")
        st.write("1: beberapa hari")
        st.write("2: lebih dari 1 minggu")
        st.write("3: hampir setiap hari")
        asrs_questions = [
            "Kesulitan Konsentrasi", "Tidak Bisa Duduk Diam", "Cepat Teralihkan", "Lupa Pekerjaan yang Harus Diselesaikan",
            "Lupa Membaca Halaman", "Kesulitan Mengatur Tugas", "Tidak Dapat Fokus Pada Detail", "Tidak Sabar",
            "Kesulitan Berbicara Tanpa Berhenti", "Merasa sulit untuk tetap fokus pada apa yang orang lain katakan", 
            "Seberapa sering Anda menyelesaikan suatu tugas tetapi melupakan rincian penting yang mengharuskan Anda mengerjakan ulang pekerjaan itu?",
            "Seberapa sering Anda merasa sulit untuk mengatur tugas atau aktivitas yang membutuhkan perencanaan sebelumnya?",
            "Seberapa sering Anda kehilangan barang-barang yang diperlukan untuk menyelesaikan tugas atau aktivitas Anda (misalnya, dokumen, kunci, telepon)",
            "Seberapa sering Anda merasa sulit untuk duduk diam dalam waktu lama, seperti saat menghadiri rapat atau menonton film?",
            "Seberapa sering Anda merasa gelisah atau merasa perlu untuk bergerak ketika Anda diminta untuk duduk diam?",
            "Seberapa sering Anda merasa kesulitan menunggu giliran Anda dalam antrean atau situasi lainnya?",
            "Seberapa sering Anda memotong pembicaraan orang lain atau menyelesaikan kalimat mereka sebelum mereka selesai berbicara?",
            "Seberapa sering Anda merasa tergesa-gesa dalam melakukan aktivitas sehari-hari?"
        ]
        asrs_scores = [st.select_slider(f"{q} (0-3):", options=[0, 1, 2, 3], key=f"asrs_{q}") 
                      for q in asrs_questions]
        asrs_total = sum(asrs_scores)
        st.write(f"ASRS Total Score: {asrs_total}")

    # AAS Tab
    with tab4:
        st.header("Adult ADHD Rating Scale (AAS)")
        st.write("0: tidak ada")
        st.write("1: ringan")
        st.write("2: sedang")
        st.write("3: berat")
        aas_questions = [
            "Kesulitan Membuat Keputusan", "Terburu-buru dalam Berbicara",
            "Lupa Meletakkan Barang", "Kesulitan Menyelesaikan Tugas",
            "Sering Tertinggal Janji", "Seberapa sering Anda merasa tidak nyaman atau bingung saat harus membaca isyarat sosial (seperti ekspresi wajah atau nada suara)?", 
            "Seberapa sering Anda merasa sulit untuk memahami lelucon, sarkasme, atau ungkapan figuratif?", "Seberapa sering Anda merasa bahwa orang lain tidak memahami cara Anda mengekspresikan diri?",
            "Seberapa sering Anda merasa sulit untuk memahami maksud orang lain ketika mereka berbicara panjang lebar?"
        ]
        aas_scores = [st.select_slider(f"{q} (0-3):", options=[0, 1, 2, 3], key=f"aas_{q}") 
                      for q in aas_questions]
        aas_total = sum(aas_scores)
        st.write(f"AAS Total Score: {aas_total}")

    # BAI Tab
    with tab5:
        st.header("Beck Anxiety Inventory (BAI)")
        st.write("0: tidak ada")
        st.write("1: ringan")
        st.write("2: sedang")
        st.write("3: berat")
        bai_questions = [
            "Kekhawatiran", "Ketegangan", "Keringat Berlebihan", "Denyut Jantung Cepat",
            "Kesulitan Bernafas", "Kelemahan", "Pusing", "Perasaan Sesak", "Gemetar", "Mual", "Mimpi Buruk", "Ketegangan Otot",
            "Seberapa sering Anda merasa terlalu sensitif terhadap suara, cahaya, atau sentuhan?",
            "Seberapa sering Anda merasa gemetar atau bergetar?", "Seberapa sering Anda merasa sulit untuk bernapas (misalnya, sesak napas atau merasa seperti kehabisan udara)?",
            "Seberapa sering Anda merasa kepala Anda pusing atau seperti akan pingsan?", "Seberapa sering Anda merasa kesemutan atau mati rasa pada bagian tubuh tertentu?", 
            "Seberapa sering Anda merasa lemas atau kehilangan energi tiba-tiba?", "Seberapa sering Anda merasa mual atau sakit perut?", "Seberapa sering Anda merasa takut sesuatu yang buruk akan terjadi?",
            "Seberapa sering Anda merasa sulit untuk berkonsentrasi karena rasa cemas?", "Seberapa sering Anda merasa bahwa tubuh Anda tegang atau kaku tanpa alasan yang jelas?"
        ]
        bai_scores = [st.select_slider(f"{q} (0-3):", options=[0, 1, 2, 3], key=f"bai_{q}") 
                      for q in bai_questions]
        bai_total = sum(bai_scores)
        st.write(f"BAI Total Score: {bai_total}")

    # Final Prediction
    st.header("Final Prediction and Treatment Recommendations")
    if st.button("Calculate ADHD Prediction"):
        input_data = np.array([[bdi_total, audit_total, aas_total, asrs_total, bai_total]])
        input_scaled = scaler.transform(input_data)

        if model_type_key == "naive_bayes":
            probability = model.predict_proba(input_scaled)[0][1]
            prediction = model.predict(input_scaled)[0]
        else:
            probability = model.predict_proba(input_scaled)[0][1]
            prediction = model.predict(input_scaled)[0]
        
        st.subheader("Results")
        st.write(f"ADHD Likelihood: {probability:.2%}")
        result = "Likely ADHD" if prediction == 1 else "Unlikely ADHD"
        st.write(f"Prediction: {result}")
        
        treatment_recommendation = get_treatment_recommendation(prediction, probability)
        st.write("Treatment Recommendation:")
        st.write(treatment_recommendation)

        # Save user input to CSV
        column_names = (
            [f"BDI_{q}" for q in range(1, len(bdi_scores)+1)] +
            [f"AUDIT_{q}" for q in range(1, len(audit_scores)+1)] +
            [f"ASRS_{q}" for q in range(1, len(asrs_scores)+1)] +
            [f"AAS_{q}" for q in range(1, len(aas_scores)+1)] +
            [f"BAI_{q}" for q in range(1, len(bai_scores)+1)] +
            ["BDI_Total", "AUDIT_Total", "ASRS_Total", "AAS_Total", "BAI_Total", 
             "ADHD_Probability", "Prediction", "Treatment_Recommendation"]
        )
        
        row_data = (bdi_scores + audit_scores + asrs_scores + aas_scores + bai_scores +
                    [bdi_total, audit_total, asrs_total, aas_total, bai_total, 
                     probability, result, treatment_recommendation])
        
        new_data = pd.DataFrame([row_data], columns=column_names)
        save_input_to_csv(new_data, DATA_FILE)

if __name__ == "__main__":
    main()