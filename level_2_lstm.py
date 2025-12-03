import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

np.random.seed(42)
tf.random.set_seed(42)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

print("Завантаження даних...")
try:
    df = pd.read_csv('reviews.csv')
except FileNotFoundError:
    print("Файл не знайдено!")
    exit()

data = df[['content', 'score']].copy()
data = data.dropna(subset=['content'])

def map_sentiment(score):
    if score in [1, 2]: return 0
    elif score in [4, 5]: return 1
    else: return np.nan

data['sentiment'] = data['score'].apply(map_sentiment)
data = data.dropna(subset=['sentiment'])
data['sentiment'] = data['sentiment'].astype(int)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

print("Очищення тексту...")
data['cleaned_content'] = data['content'].apply(clean_text)

MAX_NB_WORDS = 5000   # Максимальна кількість слів у словнику
MAX_SEQUENCE_LENGTH = 100 # Відгуки до 100 слів
EMBEDDING_DIM = 100   # Розмірність векторного представлення слова

print("Токенізація...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters=r'!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(data['cleaned_content'].values)

X = tokenizer.texts_to_sequences(data['cleaned_content'].values)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
y = data['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Розмірність X_train: {X_train.shape}")
print(f"Розмірність X_test: {X_test.shape}")

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_test.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

print("Починаємо навчання LSTM...")

history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.1, verbose=1)

print("\nОцінка на тестових даних...")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n--- Результати LSTM (Рівень 2) ---")
print(f"Точність (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

model.save('model_lstm_level2.keras')
print("Модель LSTM збережено в 'model_lstm_level2.keras'")

report_str = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])

with open("report_level_2.txt", "w", encoding="utf-8") as f:
    f.write("--- ЗВІТ ПО LSTM (RNN) ---\n\n")
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n\n")
    f.write(report_str)

print("\n[INFO] Звіт збережено у файл 'report_level_2.txt'")

def predict_lstm(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    prediction_prob = model.predict(padded)[0][0]

    label = "POSITIVE" if prediction_prob > 0.5 else "NEGATIVE"
    confidence = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob

    print(f"\nКоментар: '{text}'")
    print(f"Результат: {label} (Впевненість: {confidence:.2%})")

predict_lstm("Best app ever, five stars!")
predict_lstm("Not working, keeps crashing.")