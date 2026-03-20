"""
Model Evaluation Script for IMDB Sentiment Analysis.

This script loads a saved model, generates a classification report, plots a confusion matrix,
ROC curve, and precision-recall curve, and saves all plots to the results/visualizations/ directory.

Author: [Your Name]
Date: [Today's Date]
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.datasets import IMDBDataset
from src.model import SentimentAnalysisModel

def load_model(model_path):
    """
    Load a saved model from a file.

    Args:
        model_path (str): The path to the saved model file.

    Returns:
        SentimentAnalysisModel: The loaded model.
    """
    model = SentimentAnalysisModel()
    model.load_state_dict(torch.load(model_path))
    return model

def generate_classification_report(model, data_loader):
    """
    Generate a classification report for the model on a given dataset.

    Args:
        model (SentimentAnalysisModel): The model to evaluate.
        data_loader (DataLoader): The data loader for the dataset.

    Returns:
        str: The classification report.
    """
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return classification_report(y_true, y_pred)

def plot_confusion_matrix(model, data_loader, save_path):
    """
    Plot a confusion matrix for the model on a given dataset.

    Args:
        model (SentimentAnalysisModel): The model to evaluate.
        data_loader (DataLoader): The data loader for the dataset.
        save_path (str): The path to save the plot.
    """
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(save_path)

def plot_roc_curve(model, data_loader, save_path):
    """
    Plot an ROC curve for the model on a given dataset.

    Args:
        model (SentimentAnalysisModel): The model to evaluate.
        data_loader (DataLoader): The data loader for the dataset.
        save_path (str): The path to save the plot.
    """
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(save_path)

def plot_precision_recall_curve(model, data_loader, save_path):
    """
    Plot a precision-recall curve for the model on a given dataset.

    Args:
        model (SentimentAnalysisModel): The model to evaluate.
        data_loader (DataLoader): The data loader for the dataset.
        save_path (str): The path to save the plot.
    """
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(save_path)

def evaluate_model(model_path, data_loader):
    """
    Evaluate a model on a given dataset.

    Args:
        model_path (str): The path to the saved model file.
        data_loader (DataLoader): The data loader for the dataset.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    model = load_model(model_path)
    report = generate_classification_report(model, data_loader)
    print(report)
    cm_save_path = 'results/visualizations/confusion_matrix.png'
    roc_save_path = 'results/visualizations/roc_curve.png'
    pr_save_path = 'results/visualizations/precision_recall_curve.png'
    plot_confusion_matrix(model, data_loader, cm_save_path)
    plot_roc_curve(model, data_loader, roc_save_path)
    plot_precision_recall_curve(model, data_loader, pr_save_path)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print('F1 Score: %.4f' % f1)
    print('Accuracy: %.4f' % accuracy)
    print('Precision: %.4f' % precision)
    print('Recall: %.4f' % recall)
    return {
        'f1': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

if __name__ == '__main__':
    model_path = 'models/sentiment_analysis_model.pth'
    test_dataset = IMDBDataset('test')
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    evaluate_model(model_path, test_data_loader)
Note: This script assumes that the `IMDBDataset` class and the `SentimentAnalysisModel` class are defined in the `src.datasets` and `src.model` modules respectively. You may need to modify the script to match your actual module structure. Also, you need to make sure that the `results/visualizations/` directory exists before running the script.