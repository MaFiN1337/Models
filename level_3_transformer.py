import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Використовуємо пристрій: {device}")

print("Завантаження даних...")
try:
    df = pd.read_csv('reviews.csv')
except FileNotFoundError:
    print("Файл не знайдено!")
    exit()

data = df[['content', 'score']].copy()
data = data.dropna(subset=['content'])

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

data = data.sample(n=2000, random_state=42)
print(f"Для швидкості використаємо випадкові {len(data)} відгуків.")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    data['content'], data['sentiment'], test_size=0.2, random_state=42
)

print("Завантажуємо DistilBERT Tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


train_dataset = ReviewDataset(X_train_text, y_train, tokenizer)
test_dataset = ReviewDataset(X_test_text, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

print("Завантажуємо модель DistilBERT...")
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 3
print(f"Починаємо навчання на {epochs} епох...")

for epoch in range(epochs):
    print(f"--- Епоха {epoch + 1}/{epochs} ---")
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"Середній Loss за епоху: {avg_train_loss:.4f}")

print("\nОцінка моделі...")
model.eval()

predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

        predictions.extend(preds.cpu().tolist())
        true_labels.extend(labels.cpu().tolist())

print("\n--- Результати Transformer (Рівень 3) ---")
print(f"Точність (Accuracy): {accuracy_score(true_labels, predictions):.4f}")
print("\nClassification Report:")
print(classification_report(true_labels, predictions, target_names=['Negative', 'Positive']))

torch.save(model.state_dict(), 'model_bert_level3.pth')


def predict_bert(text):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    label = "POSITIVE" if pred_class == 1 else "NEGATIVE"
    confidence = probs[0][pred_class].item()

    print(f"\nКоментар: '{text}'")
    print(f"Результат: {label} (Впевненість: {confidence:.2%})")

print("\n--- Перевірка DistilBERT ---")
predict_bert("Usefull app, helped me a lot")
predict_bert("I hate this layout, very confusing")