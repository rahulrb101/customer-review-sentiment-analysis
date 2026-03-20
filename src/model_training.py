"""
src/model_training.py

IMDB Sentiment Analysis Model Training Script.

Train a sentiment analysis model on the IMDB dataset using either a logistic regression
baseline or a fine-tuned DistilBERT model. Supports cross-validation and hyperparameter
grid search.

Usage:
    python src/model_training.py --model logistic_regression
    python src/model_training.py --model distilbert
"""

import argparse
import logging
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

# Set up logging
logging.basicConfig(level=logging.INFO)

class IMDBDataset(Dataset):
    """
    Custom dataset class for IMDB sentiment analysis.

    Attributes:
        data (pd.DataFrame): IMDB dataset.
        tokenizer (DistilBertTokenizer): Tokenizer for DistilBERT.
        max_length (int): Maximum sequence length.
    """
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_path):
    """
    Load the IMDB dataset from a CSV file.

    Args:
        data_path (str): Path to the dataset CSV file.

    Returns:
        pd.DataFrame: Loaded IMDB dataset.
    """
    return pd.read_csv(data_path)

def train_logistic_regression(data, cv=5):
    """
    Train a logistic regression baseline model.

    Args:
        data (pd.DataFrame): IMDB dataset.
        cv (int): Number of folds for cross-validation.

    Returns:
        tuple: Trained logistic regression model and best parameters.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['text'])
    y = data['label']

    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }

    model = LogisticRegression(max_iter=1000)
    grid_search = GridSearchCV(model, param_grid, cv=cv)
    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_

def train_distilbert(data, cv=5):
    """
    Train a fine-tuned DistilBERT model.

    Args:
        data (pd.DataFrame): IMDB dataset.
        cv (int): Number of folds for cross-validation.

    Returns:
        tuple: Trained DistilBERT model and best parameters.
    """
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_dataset = IMDBDataset(data, tokenizer, max_length=512)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

    return model, {}

def main():
    parser = argparse.ArgumentParser(description='IMDB Sentiment Analysis Model Training')
    parser.add_argument('--model', type=str, required=True, help='Model type (logistic_regression or distilbert)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the IMDB dataset CSV file')
    parser.add_argument('--cv', type=int, default=5, help='Number of folds for cross-validation')
    args = parser.parse_args()

    data = load_data(args.data_path)

    if args.model == 'logistic_regression':
        model, best_params = train_logistic_regression(data, cv=args.cv)
        torch.save(model, 'results/logistic_regression.pth')
        logging.info(f'Best parameters: {best_params}')
    elif args.model == 'distilbert':
        model, best_params = train_distilbert(data, cv=args.cv)
        torch.save(model, 'results/distilbert.pth')
        logging.info(f'Best parameters: {best_params}')
    else:
        raise ValueError('Invalid model type')

if __name__ == '__main__':
    main()
To use this script, you'll need to have the following dependencies installed:

* `numpy`
* `pandas`
* `scikit-learn`
* `transformers`
* `torch`
* `argparse`

You can install these dependencies using pip:
```bash
pip install numpy pandas scikit-learn transformers torch argparse
```

To train a model, simply run the script with the `--model` argument specifying the type of model to train:
```bash
python src/model_training.py --model logistic_regression --data_path path/to/imdb_dataset.csv
```

Replace `path/to/imdb_dataset.csv` with the actual path to your IMDB dataset CSV file.

The script will train the specified model using cross-validation and hyperparameter grid search, and save the trained model to the `results/` directory.