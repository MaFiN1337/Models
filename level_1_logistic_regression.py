import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

try:
    df = pd.read_csv('reviews.csv')
    print("Файл 'reviews.csv' успішно завантажено.")
except FileNotFoundError:
    print("ПОМИЛКА: Файл 'reviews.csv' не знайдено.")
    exit()

data = df[['content', 'score']].copy()


def map_sentiment(score):
    if score in [1, 2]:
        return 0
    elif score in [4, 5]:
        return 1
    else:
        return np.nan

data['sentiment'] = data['score'].apply(map_sentiment)

data = data.dropna(subset=['sentiment'])
data['sentiment'] = data['sentiment'].astype(int)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return ' '.join(words)

print("Починаємо очищення тексту... (це може зайняти хвилину)")
data['cleaned_content'] = data['content'].apply(clean_text)
print("Очищення завершено.")

X = data['cleaned_content']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Загальний розмір вибірки (після очищення): {len(data)}")
print(f"Навчання: {len(X_train)}, Тест: {len(X_test)}")

print("Проводимо TF-IDF векторизацію...")
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print(f"Форма TF-IDF матриці (трейн): {X_train_tfidf.shape}")

print("Навчаємо Логістичну Регресію...")
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train_tfidf, y_train)
print("Навчання завершено.")

y_pred = model_lr.predict(X_test_tfidf)

print("\n" + "=" * 40)
print("--- Результати Логістичної Регресії (Рівень 1) ---")
print(f"Точність (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)']))
print("=" * 40)

joblib.dump(model_lr, 'model_lr_level1.pkl')
joblib.dump(vectorizer, 'vectorizer_level1.pkl')

report_str = classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)'])

with open("report_level_1.txt", "w", encoding="utf-8") as f:
    f.write("--- ЗВІТ ПО ЛОГІСТИЧНІЙ РЕГРЕСІЇ ---\n\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n\n")
    f.write(report_str)

print("\n[INFO] Звіт збережено у файл 'report_level_1.txt'")

def predict_single_review(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred_class = model_lr.predict(vec)[0]

    label = "POSITIVE" if pred_class == 1 else "NEGATIVE"
    print(f"\nКоментар: '{text}'")
    print(f"Результат: {label}")

predict_single_review("This app is absolutely amazing, I use it every day!")
predict_single_review("Total waste of time, too many bugs.")
predict_single_review("It's okay, but needs update.")