# Customer Review Sentiment Analysis using IMDB 50K dataset
[![Python](https://img.shields.io/badge/Python-3.9-yellow.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/rahulrb101/customer-review-sentiment-analysis?style=social)](https://github.com/rahulrb101/customer-review-sentiment-analysis)

## Project Description
This project aims to develop a sentiment analysis model to classify customer reviews from the IMDB 50K dataset as positive or negative. The model can be used in various industries, including finance, analytics, and consulting, to analyze customer feedback and improve customer satisfaction.

## Features
* **Text Preprocessing**: Tokenization, stopword removal, stemming, and lemmatization
* **Feature Extraction**: Bag-of-words, TF-IDF, and word embeddings (Word2Vec, GloVe)
* **Model Selection**: Logistic Regression, Decision Trees, Random Forest, Support Vector Machines, and Convolutional Neural Networks (CNN)
* **Hyperparameter Tuning**: Grid search and random search for optimal hyperparameters
* **Model Evaluation**: Accuracy, F1-score, precision, and recall

## Project Structure
customer-review-sentiment-analysis/
├── data/
│   ├── IMDB_Dataset.csv
│   └── preprocessing.py
├── models/
│   ├── logistic_regression.py
│   ├── decision_trees.py
│   ├── random_forest.py
│   ├── svm.py
│   └── cnn.py
├── utils/
│   ├── feature_extraction.py
│   └── hyperparameter_tuning.py
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
## Installation
To install the required libraries, run the following command:
```bash
pip install -r requirements.txt
```
Make sure to install the necessary packages, including `numpy`, `pandas`, `scikit-learn`, `tensorflow`, and `keras`.

## Usage Examples
To run the model, use the following command:
```bash
python main.py --model logistic_regression --data IMDB_Dataset.csv
```
Replace `logistic_regression` with the desired model and `IMDB_Dataset.csv` with the dataset file.

## Model Results
The following table shows the model performance on the test dataset:
| Model | Accuracy | F1-score |
| --- | --- | --- |
| Logistic Regression | 0.85 | 0.83 |
| Decision Trees | 0.82 | 0.81 |
| Random Forest | 0.88 | 0.86 |
| Support Vector Machines | 0.84 | 0.83 |
| Convolutional Neural Networks | 0.90 | 0.89 |

## Visualizations
The following visualizations provide insight into the model performance:
* **Confusion Matrix**: shows true positives, false positives, true negatives, and false negatives
* **ROC Curve**: plots the true positive rate against the false positive rate
* **Precision-Recall Curve**: plots precision against recall

## Dataset Information
The IMDB 50K dataset consists of 50,000 movie reviews from IMDB, with 25,000 positive and 25,000 negative reviews.

## Tech Stack
* **Programming Language**: Python 3.9
* **Libraries**: NumPy, pandas, scikit-learn, TensorFlow, Keras
* **Frameworks**: scikit-learn for machine learning, TensorFlow and Keras for deep learning

## Contributing
To contribute to this project, please follow these steps:
1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Commit your changes with a descriptive message
4. Push your changes to the new branch
5. Create a pull request

## License
This project is licensed under the MIT License.
