This script `src/data_preprocessing.py` will handle loading the IMDB 50K dataset, cleaning the text data using NLTK (removing stopwords, punctuation, lemmatization), splitting it into training and testing sets, and saving the processed data.

**Before running, ensure you have the necessary NLTK data downloaded.** The script will attempt to download them automatically if they are missing, but it's good to be aware.

# src/data_preprocessing.py

import pandas as pd
import numpy as np
import re
import string
import os
import logging
from typing import Tuple, Dict

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split

# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- Constants and Configuration ---
# Define paths and filenames
DATA_DIR = 'data'
RAW_DATA_PATH = os.path.join(DATA_DIR, 'IMDB Dataset.csv')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_data.csv')
TEST_OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, 'test_data.csv')

# Column names
TEXT_COLUMN = 'review'
LABEL_COLUMN = 'sentiment'
CLEANED_TEXT_COLUMN = 'cleaned_review'

# Train-test split parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- NLTK Setup ---
def nltk_setup() -> None:
    """
    Downloads necessary NLTK data packages if they are not already present.
    """
    required_packages = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4' # Open Multilingual Wordnet, dependency for wordnet
    }
    
    for package_name, package_id in required_packages.items():
        try:
            nltk.data.find(package_id)
            logger.info(f"NLTK package '{package_name}' already downloaded.")
        except nltk.downloader.DownloadError:
            logger.info(f"Downloading NLTK package '{package_name}'...")
            nltk.download(package_name)
            logger.info(f"NLTK package '{package_name}' downloaded successfully.")
        except Exception as e:
            logger.error(f"Error checking/downloading NLTK package '{package_name}': {e}")
            raise

# --- Data Loading Function ---
def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV dataset from the specified file path into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        pd.errors.ParserError: If the CSV file is malformed.
    """
    logger.info(f"Attempting to load data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: The file '{file_path}' was not found.")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Error: The file '{file_path}' is empty.")
        raise
    except pd.errors.ParserError:
        logger.error(f"Error: Could not parse the CSV file '{file_path}'. It might be malformed.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}")
        raise

# --- Text Cleaning Function ---
def clean_text(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    """
    Performs a series of text cleaning steps:
    1. Converts text to lowercase.
    2. Removes HTML tags (e.g., <br />).
    3. Removes URLs.
    4. Removes special characters, numbers, and punctuation.
    5. Tokenizes the text.
    6. Removes stopwords.
    7. Lemmatizes the words.
    8. Joins the cleaned words back into a string.

    Args:
        text (str): The input text string to be cleaned.
        stop_words (set): A set of NLTK stopwords.
        lemmatizer (WordNetLemmatizer): An initialized NLTK WordNetLemmatizer.

    Returns:
        str: The cleaned and preprocessed text string.
    """
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 4. Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 5. Tokenize text
    tokens = word_tokenize(text)

    # 6. Remove stopwords and empty tokens
    tokens = [word for word in tokens if word not in stop_words and word.strip()]

    # 7. Lemmatize words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 8. Join back to string
    cleaned_text = ' '.join(tokens)
    
    # Remove extra spaces that might result from cleaning
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

# --- Preprocessing Dataset Function ---
def preprocess_dataset(df: pd.DataFrame, text_column: str, cleaned_text_column: str) -> pd.DataFrame:
    """
    Applies the text cleaning function to a specified text column in the DataFrame
    and creates a new column for the cleaned text.

    Args:
        df (pd.DataFrame): The input DataFrame containing the raw text.
        text_column (str): The name of the column containing the raw text reviews.
        cleaned_text_column (str): The name of the new column to store the cleaned text.

    Returns:
        pd.DataFrame: The DataFrame with an additional column for cleaned text.
    """
    if text_column not in df.columns:
        logger.error(f"Error: Text column '{text_column}' not found in DataFrame.")
        raise ValueError(f"Text column '{text_column}' not found.")

    logger.info(f"Starting text preprocessing for column '{text_column}'...")
    
    # Initialize NLTK components once for efficiency
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Apply the clean_text function
    df[cleaned_text_column] = df[text_column].apply(
        lambda x: clean_text(x, stop_words, lemmatizer)
    )
    logger.info(f"Text preprocessing completed. New column '{cleaned_text_column}' created.")
    
    return df

# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("Starting data preprocessing script...")

    # 1. Setup NLTK
    try:
        nltk_setup()
    except Exception as e:
        logger.error(f"Failed to set up NLTK: {e}")
        exit(1)

    # 2. Load the dataset
    try:
        imdb_df = load_data(RAW_DATA_PATH)
    except Exception:
        logger.error("Exiting due to data loading error.")
        exit(1)

    # Convert sentiment labels to numerical (optional, but good for ML models)
    # 'positive' -> 1, 'negative' -> 0
    imdb_df[LABEL_COLUMN] = imdb_df[LABEL_COLUMN].map({'positive': 1, 'negative': 0})
    logger.info("Converted 'sentiment' column to numerical: 'positive'=1, 'negative'=0.")
    logger.info(f"Sentiment distribution: \n{imdb_df[LABEL_COLUMN].value_counts()}")

    # 3. Preprocess the text data
    try:
        processed_df = preprocess_dataset(imdb_df, TEXT_COLUMN, CLEANED_TEXT_COLUMN)
    except ValueError as e:
        logger.error(f"Exiting due to preprocessing error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during preprocessing: {e}")
        exit(1)

    # Check for empty cleaned reviews after processing
    empty_reviews_count = processed_df[processed_df[CLEANED_TEXT_COLUMN].apply(lambda x: not x.strip())].shape[0]
    if empty_reviews_count > 0:
        logger.warning(f"{empty_reviews_count} reviews resulted in empty strings after cleaning. Consider reviewing text cleaning steps.")
        # Optionally, remove rows with empty cleaned reviews
        # processed_df = processed_df[processed_df[CLEANED_TEXT_COLUMN].apply(lambda x: x.strip())]
        # logger.info(f"Removed {empty_reviews_count} rows with empty cleaned reviews. Remaining shape: {processed_df.shape}")

    # 4. Split the dataset into training and testing sets
    logger.info(f"Splitting data into training ({1-TEST_SIZE:.0%}) and testing ({TEST_SIZE:.0%}) sets...")
    X = processed_df[CLEANED_TEXT_COLUMN]
    y = processed_df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Reconstruct dataframes for saving
    train_df = pd.DataFrame({CLEANED_TEXT_COLUMN: X_train, LABEL_COLUMN: y_train})
    test_df = pd.DataFrame({CLEANED_TEXT_COLUMN: X_test, LABEL_COLUMN: y_test})

    logger.info(f"Training set shape: {train_df.shape}")
    logger.info(f"Testing set shape: {test_df.shape}")
    logger.info(f"Training sentiment distribution: \n{train_df[LABEL_COLUMN].value_counts(normalize=True)}")
    logger.info(f"Testing sentiment distribution: \n{test_df[LABEL_COLUMN].value_counts(normalize=True)}")

    # 5. Save the preprocessed datasets
    logger.info("Saving preprocessed training and testing data...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    try:
        train_df.to_csv(TRAIN_OUTPUT_PATH, index=False)
        logger.info(f"Training data saved to: {TRAIN_OUTPUT_PATH}")

        test_df.to_csv(TEST_OUTPUT_PATH, index=False)
        logger.info(f"Testing data saved to: {TEST_OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Error saving preprocessed data: {e}")
        exit(1)

    logger.info("Data preprocessing completed successfully.")
**To run this script:**

1.  **Project Structure:**
    Ensure your project structure looks like this:

    ```
    sentiment_analysis/
    ├── src/
    │   └── data_preprocessing.py
    └── data/
        └── IMDB Dataset.csv
    ```
    *   Place the `IMDB Dataset.csv` (which typically contains `review` and `sentiment` columns) inside the `data/` directory. If you don't have it, you can download it from Kaggle: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

2.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn nltk
    ```

3.  **Execute the Script:**
    Navigate to the `sentiment_analysis/` directory in your terminal and run:
    ```bash
    python src/data_preprocessing.py
    ```

**Output:**

The script will:
*   Download necessary NLTK data (if not already present).
*   Load the `IMDB Dataset.csv`.
*   Convert 'positive'/'negative' sentiment to 1/0.
*   Clean the text reviews (lowercase, remove HTML, URLs, numbers, punctuation, stopwords, lemmatize).
*   Split the dataset into `train_data.csv` and `test_data.csv` (80/20 split, stratified by sentiment).
*   Save these two new CSV files into `data/processed/`.

You will see logging messages indicating the progress and any issues.