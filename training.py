import pandas as pd
import re
import string
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# =========================
# 1. LOAD DATA
# =========================
def load_data(path):
    df = pd.read_csv(path, sep=';')
    df.columns = df.columns.str.strip().str.lower()

    print("\nKolom dataset:", df.columns)
    return df


# =========================
# 2. PREPROCESSING
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text


# =========================
# 3. TRAIN MODEL SENTIMEN
# =========================
def train_sentiment(df):
    print("\n=== TRAINING MODEL SENTIMEN ===")

    X = df['clean_text']
    y = df['sentimen']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    # =========================
    # EVALUASI MODEL
    # =========================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n=== Evaluasi Sentimen ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print("\nDetail Klasifikasi:")
    print(classification_report(y_test, y_pred))

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model_sentimen.pkl')
    joblib.dump(tfidf, 'models/tfidf_sentimen.pkl')


# =========================
# 4. TRAIN MODEL KATEGORI
# =========================
def train_category(df):
    print("\n=== TRAINING MODEL KATEGORI ===")

    X = df['clean_text']
    y = df['kategori']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer(max_features=5000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    # =========================
    # EVALUASI MODEL
    # =========================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n=== Evaluasi Kategori ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print("\nDetail Klasifikasi:")
    print(classification_report(y_test, y_pred))

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model_kategori.pkl')
    joblib.dump(tfidf, 'models/tfidf_kategori.pkl')


# =========================
# 5. MAIN
# =========================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'Dataset_berhasil1.csv')

    df = load_data(DATA_PATH)

    # validasi kolom
    required_columns = ['content', 'sentimen', 'kategori']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset!")

    # preprocessing
    df['clean_text'] = df['content'].apply(clean_text)

    # training
    train_sentiment(df)
    train_category(df)

    print("\n✅ Training selesai! Model tersimpan di folder 'models/'")