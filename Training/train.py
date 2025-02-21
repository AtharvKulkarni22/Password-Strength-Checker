import pandas as pd
import numpy as np
import re
import string
import joblib
import math
import random
import matplotlib.pyplot as plt
from collections import Counter
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.corpus import words

from features import (
    password_length,
    contains_digit,
    contains_special,
    contains_upper,
    word_match,
    levenshtein_similarity,
    character_entropy,
    contains_keyboard_pattern,
    contains_repeated_chars,
    vowel_to_consonant_ratio,
    extract_engineered_features
)

def main():
    nltk.download("words")

    # Load Dataset
    print("Loading dataset")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "Dataset", "cleaned_data.csv")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded with shape: {df.shape}")

    df.dropna(inplace=True)

    # Split Dataset into Train (70%), Dev (20%), Test (10%)
    print("Splitting dataset into Train, Dev, and Test sets")
    train_data, temp_data = train_test_split(df, test_size=0.3, random_state=10)
    dev_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=10)
    print(f"Train shape: {train_data.shape}, Dev shape: {dev_data.shape}, Test shape: {test_data.shape}")

    # Apply Feature Engineering to Train, Dev, and Test Data
    def extract_features(df):
        print("Extracting features for the dataset")
        df["length"] = df["password"].apply(password_length)
        df["has_digit"] = df["password"].apply(contains_digit).astype(int)
        df["has_special"] = df["password"].apply(contains_special).astype(int)
        df["has_upper"] = df["password"].apply(contains_upper).astype(int)
        df["is_word_match"] = df["password"].apply(word_match).astype(int)
        df["levenshtein_distance"] = df["password"].apply(levenshtein_similarity)
        df["entropy"] = df["password"].apply(character_entropy)
        df["has_keyboard_pattern"] = df["password"].apply(contains_keyboard_pattern).astype(int)
        df["has_repeated_chars"] = df["password"].apply(contains_repeated_chars).astype(int)
        df["vowel_consonant_ratio"] = df["password"].apply(vowel_to_consonant_ratio)
        print("Feature extraction complete. New columns added:", df.columns.tolist()[2:])
        return df

    print("Applying feature extraction")
    train_data = extract_features(train_data)
    dev_data = extract_features(dev_data)
    test_data = extract_features(test_data)

    # TF-IDF Vectorization
    print("Performing TF-IDF vectorization on password text")
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=10000)
    X_tfidf_train = tfidf.fit_transform(train_data["password"])
    X_tfidf_dev = tfidf.transform(dev_data["password"])
    X_tfidf_test = tfidf.transform(test_data["password"])
    print("TF-IDF vectorization complete.")
    print(f"TF-IDF shape - Train: {X_tfidf_train.shape}, Dev: {X_tfidf_dev.shape}, Test: {X_tfidf_test.shape}")

    # Combine TF-IDF with Extracted Features
    print("Combining TF-IDF features with extracted features")
    X_train = np.hstack((X_tfidf_train.toarray(), train_data.iloc[:, 2:].values))
    X_dev = np.hstack((X_tfidf_dev.toarray(), dev_data.iloc[:, 2:].values))
    X_test = np.hstack((X_tfidf_test.toarray(), test_data.iloc[:, 2:].values))
    print("Combined feature shapes - X_train:", X_train.shape, ", X_dev:", X_dev.shape, ", X_test:", X_test.shape)

    y_train = train_data["strength"].values
    y_dev = dev_data["strength"].values
    y_test = test_data["strength"].values

    print("Initialize SGDClassifier")
    model = SGDClassifier(random_state=10)
    classes = np.unique(y_train)  # [0, 1, 2]
    print("Model initialized.")

    # Training Loop with Dev Set Evaluation
    batch_size = 1000
    dev_sample_size = 1000
    dev_accuracies = []

    print("Starting training loop over batches")
    total_batches = math.ceil(len(X_train) / batch_size)
    for batch_idx, i in enumerate(range(0, len(X_train), batch_size), 1):
        print(f"\nProcessing batch {batch_idx}/{total_batches} (indices {i} to {i+batch_size})...")
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # Incremental Training:
        if batch_idx == 1:
            model.partial_fit(X_batch, y_batch, classes=classes)
        else:
            model.partial_fit(X_batch, y_batch)
        print("Batch training complete.")
        
        # Evaluate on a random sample from the Dev Set
        print("Evaluating model on a dev sample")
        dev_sample_indices = np.random.choice(len(X_dev), dev_sample_size, replace=False)
        X_dev_sample = X_dev[dev_sample_indices]
        y_dev_sample = y_dev[dev_sample_indices]
        
        y_dev_pred = model.predict(X_dev_sample)
        dev_acc = accuracy_score(y_dev_sample, y_dev_pred)
        dev_accuracies.append(dev_acc)
        print(f"Dev set accuracy after batch {batch_idx}: {dev_acc:.4f}")

    print("\nTraining complete.")

    # Final Testing on Test Set
    print("Performing final evaluation on test set")
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # Save Model
    model_dir = os.path.join(base_dir, "..", "Models")
    os.makedirs(model_dir, exist_ok=True)
    print("Saving model")
    joblib.dump(model, os.path.join(model_dir, "password_strength_model.pkl"))
    joblib.dump(tfidf, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    print("Model and vectorizer saved to:", model_dir)

    # Plot Dev Set Accuracy Over Time
    print("Plotting dev set accuracy over training batches")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(dev_accuracies)+1), dev_accuracies, marker='o', linestyle='-')
    plt.xlabel("Training Batch Number")
    plt.ylabel("Dev Set Accuracy")
    plt.title("Model Accuracy on Dev Set During Training")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
