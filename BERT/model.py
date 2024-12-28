# Importing Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset

# Parameters for BERT model
bert_path = "bert-base-uncased"
max_seq_length = 256

# Concatenating the two datasets
#C:\Users\mehta\Sarcasm-Detector-bert-lstm\Dataset\Sarcasm_Headlines_Dataset.json
df1 = pd.read_json(r"C:\Users\mehta\Sarcasm-Detector-bert-lstm\Dataset\Sarcasm_Headlines_Dataset.json", lines=True)
df2 = pd.read_json(r"C:\Users\mehta\Sarcasm-Detector-bert-lstm\Dataset\Sarcasm_Headlines_Dataset_v2.json", lines=True)
df = pd.concat([df1, df2])

text = df['headline'].tolist()
text = [' '.join(t.split()[0:max_seq_length]) for t in text]
text_label = df['is_sarcastic'].tolist()

train_text, test_text, train_labels, test_labels = train_test_split(text, text_label, random_state=0)

# Creating a custom Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_seq_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Instantiate tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_path)

# Convert data to Dataset format
train_dataset = TextDataset(train_text, train_labels, tokenizer, max_seq_length)
test_dataset = TextDataset(test_text, test_labels, tokenizer, max_seq_length)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# Define the BERT model
class BertClassifier(nn.Module):
    def __init__(self, bert_path, n_classes=1):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output  # Use pooler_output instead of tuple unpacking
        output = self.dropout(pooled_output)
        output = self.fc(output)
        return self.sigmoid(output)

# Instantiate the model
model = BertClassifier(bert_path)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(10):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device).float()

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Evaluating the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device).float()

        outputs = model(input_ids, attention_mask, token_type_ids)
        predicted = (outputs.squeeze() > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy: {accuracy}')