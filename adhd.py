import streamlit as st
import numpy as np
import pandas as pd
import os
from collections import Counter

# Custom ML Implementations
class CustomKNN:
    def __init__(self, k=5):
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
    def __init__(self):
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
        return np.prod(1 / np.sqrt(2 * np.pi * var) * exponent)  # Removed axis=1

    
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
                probabilities.append([1/len(self.classes)] * len(self.classes))
            else:
                probabilities.append([p/total for p in likelihoods])
                
        return np.array(probabilities)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def train_test_split(X, test_size):
    indices = np.random.permutation(len(X))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx]

class CustomScaler:
    def __init__(self):
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
    base_dir = os.path.dirname(__file__)  # Mendapatkan direktori skrip yang dijalankan
    full_file_path = os.path.join(base_dir, file_path)
    # Load dataset
    if not os.path.exists(full_file_path):
        raise FileNotFoundError(f"Dataset '{file_path}' tidak ditemukan.")
    
    data = pd.read_excel(full_file_path)
    
    # Pilih kolom yang relevan untuk pelatihan (input features dan target label)
    feature_columns = [
        'bdi1_total', 'audit1_total', 'aas1_total', 'asrs1_total.x', 'bai1_total'
    ]
    target_column = 'adhd_label'
    
    X = data[feature_columns].values
    y = data[target_column].values
    
    # Split data into training and testing sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Scale data
    scaler = CustomScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    if model_type == "naive_bayes":
        model = CustomGaussianNB()
    else:
        model = CustomKNN(k=5)
    
    model.fit(X_train_scaled, y_train)
    return model, scaler

def save_input_to_csv(data, file_path):
    if not os.path.exists(file_path):
        data.to_csv(file_path, index=False)
    else:
        existing_data = pd.read_csv(file_path)
        updated_data = pd.concat([existing_data, data], ignore_index=True)
        updated_data.to_csv(file_path, index=False)

# Main Application
def main():
    st.set_page_config(page_title="ADHD Prediction App", layout="wide")
    
    DATA_FILE = "adhd_data.csv"
    
    st.title("ADHD Prediction System")
    st.write("Please fill out the following questionnaires to assess ADHD likelihood.")
    
    # Model selection
    model_type = st.sidebar.selectbox("Select Prediction Model", ["Naive Bayes", "K-Nearest Neighbors"])
    model_type_key = "naive_bayes" if model_type == "Naive Bayes" else "knn"
    
    model, scaler = load_model_and_scaler(model_type_key)
    
    # Create tabs for different questionnaires
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["BDI", "AUDIT", "ASRS", "AAS", "BAI"])
    
    # BDI Tab
    with tab1:
        st.header("Beck Depression Inventory (BDI)")
        st.write("0: tidak ada")
        st.write("1: ringan")
        st.write("2: sedang")
        st.write("3: berat")
        bdi_questions = [
            "Kesedihan", "Pesimisme", "Kegagalan Masa Lalu", "Kehilangan Kesenangan", "Rasa Bersalah", "Perasaan Hukuman", "Tidak Suka Diri Sendiri", "Sikap Mengkritik Diri Sendiri", "Pikiran Bunuh Diri", "Menangis", "Agitasi", "Kehilangan Minat", "Keragu-raguan", "Tidak Berharga", "Kehilangan Energi", "Perubahan Tidur", "Mudah Tersinggung", "Perubahan Nafsu Makan", "Kesulitan Konsentrasi", "Kelelahan"
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
            "Frekuensi Minum", "Jumlah Konsumsi yang Biasa",
            "Frekuensi Episode Minum Berat", "Kontrol yang Terganggu terhadap Minum",
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
        st.write("1: jarang")
        st.write("2: kadang-kadang")
        st.write("3: sering")
        st.write("4: sangat sering")
        asrs_questions = [
            "Kesulitan memperhatikan detail", "Kesulitan mempertahankan perhatian",
            "Kesulitan mendengarkan", "Tidak menyelesaikan tugas", "Kesulitan mengatur tugas",
            "Menghindari tugas mental", "Sering kehilangan barang", "Mudah terganggu",
            "Sering lupa aktivitas", "Gelisah", "Kesulitan duduk diam", 
            "Merasa gelisah fisik", "Selalu sibuk", "Berbicara berlebihan",
            "Membuat jawaban sebelum pertanyaan selesai", "Kesulitan menunggu giliran",
            "Menyela orang lain", "Tidak dapat menahan keinginan"
        ]
        asrs_scores = [st.select_slider(f"{q} (0-4):", options=[0, 1, 2, 3, 4], key=f"asrs_{q}") 
                      for q in asrs_questions]
        asrs_total = sum(asrs_scores)
        st.write(f"ASRS Total Score: {asrs_total}")

    # AAS Tab
    with tab4:
        st.header("Anxiety Assessment Scale (AAS)")
        st.write("1: tidak pernah")
        st.write("2: jarang")
        st.write("3: kadang-kadang")
        st.write("4: sering")
        st.write("5: sangat sering")
        aas_questions = [
            "Merasakan mati rasa", "Merasa panas tiba-tiba", "Merasa lemah",
            "Rasa gemetar", "Merasa takut", "Kesulitan bernapas", "Merasa tercekik",
            "Jantung berdebar", "Merasa gugup", "Merasa takut kehilangan kendali",
            "Merasa gemetar di bagian tubuh", "Masalah pencernaan", "Merasa pusing",
            "Merasa tegang", "Sulit untuk santai", "Takut tubuh gagal berfungsi",
            "Wajah memerah", "Merasa berkeringat", "Merasa mudah terkejut",
            "Merasa kesulitan fokus", "Sulit tidur karena cemas"
        ]
        aas_scores = [st.select_slider(f"{q} (1-5):", options=[1, 2, 3, 4, 5], key=f"aas_{q}") 
                     for q in aas_questions]
        aas_total = sum(aas_scores)
        st.write(f"AAS Total Score: {aas_total}")

    # BAI Tab
    with tab5:
        st.header("Beck Anxiety Inventory (BAI)")
        st.write("0: tidak sama sekali")
        st.write("1: ringan")
        st.write("2: sedang")
        st.write("3: parah")
        bai_questions = [
            "Merasa kesemutan", "Merasa lemas", "Merasa pusing",
            "Merasa gemetar", "Detak jantung cepat", "Ketegangan otot",
            "Takut kehilangan kendali", "Sesak napas", "Merasa gugup",
            "Takut hal buruk terjadi", "Merasa panas", "Mual",
            "Merasa tercekik", "Sulit fokus", "Wajah memerah",
            "Takut ruang terbuka", "Berkeringat berlebihan", "Merasa tidak nyata",
            "Kehilangan keseimbangan", "Takut pingsan", "Takut mati"
        ]
        bai_scores = [st.select_slider(f"{q} (0-5):", options=[0, 1, 2, 3, 4], key=f"bai_{q}") 
                     for q in bai_questions]
        bai_total = sum(bai_scores)
        st.write(f"BAI Total Score: {bai_total}")

    # Prediction Section
    st.header("Prediction")
    if st.button("Calculate ADHD Prediction"):
        input_data = np.array([[bdi_total, audit_total, aas_total, asrs_total, bai_total]])
        input_scaled = scaler.transform(input_data)
        
        # Get prediction and probability
        if model_type_key == "naive_bayes":
            probability = model.predict_proba(input_scaled)[0][1]
            prediction = model.predict(input_scaled)[0]
        else:
            probability = model.predict_proba(input_scaled)[0][1]
            prediction = model.predict(input_scaled)[0]
        
        # Display results
        st.subheader("Results")
        st.write(f"ADHD Likelihood: {probability:.2%}")
        
        result = "Likely ADHD" if prediction == 1 else "Unlikely ADHD"
        st.write(f"Prediction: {result}")
        
        # Prepare data for saving
        column_names = (
            [f"BDI_{q}" for q in range(1, len(bdi_scores)+1)] +
            [f"AUDIT_{q}" for q in range(1, len(audit_scores)+1)] +
            [f"ASRS_{q}" for q in range(1, len(asrs_scores)+1)] +
            [f"AAS_{q}" for q in range(1, len(aas_scores)+1)] +
            [f"BAI_{q}" for q in range(1, len(bai_scores)+1)] +
            ["BDI_Total", "AUDIT_Total", "ASRS_Total", "AAS_Total", "BAI_Total", 
             "ADHD_Probability", "Prediction"]
        )
        
        row_data = (bdi_scores + audit_scores + asrs_scores + aas_scores + bai_scores +
                   [bdi_total, audit_total, asrs_total, aas_total, bai_total, 
                    probability, result])
        
        new_data = pd.DataFrame([row_data], columns=column_names)
        save_input_to_csv(new_data, DATA_FILE)

if __name__ == "__main__":
    main()
