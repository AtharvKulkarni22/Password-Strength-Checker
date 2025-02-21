import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
import os
import sys
import nltk
from nltk.corpus import words

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
training_dir = os.path.join(project_root, "Training")
if training_dir not in sys.path:
    sys.path.append(training_dir)

from features import extract_engineered_features

nltk.download("words")
english_words = set(words.words())

model_path = os.path.join(project_root, "Models", "password_strength_model.pkl")
tfidf_path = os.path.join(project_root, "Models", "tfidf_vectorizer.pkl")

try:
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
except Exception as e:
    raise RuntimeError("Error loading model or vectorizer: " + str(e))

def rate_password(password):
    tfidf_features = tfidf.transform([password]).toarray()
    
    engineered_features = extract_engineered_features(password)
    
    input_features = np.hstack((tfidf_features, engineered_features))
    
    pred = model.predict(input_features)[0]
    
    if pred == 0:
        rating = "Weak"
    elif pred == 1:
        rating = "Medium"
    elif pred == 2:
        rating = "Strong"
    else:
        rating = "Unknown"
    return rating

def check_password():
    password = entry.get()
    if not password:
        messagebox.showerror("Input Error", "Please enter a password!")
        return
    rating = rate_password(password)
    
    if rating == "Weak":
        color = "red"
    elif rating == "Medium":
        color = "orange"
    elif rating == "Strong":
        color = "green"
    else:
        color = "black"
        
    result_label.config(text=f"Password Strength: {rating}", fg=color)

root = tk.Tk()
root.title("Password Strength Checker")
root.geometry("600x300")
root.resizable(False, False)

instruction_label = tk.Label(root, text="Enter your password:", font=("Helvetica", 14))
instruction_label.pack(pady=15)

entry = tk.Entry(root, width=50, font=("Helvetica", 14))
entry.pack(pady=5)

check_button = tk.Button(root, text="Check Strength", command=check_password, font=("Helvetica", 14))
check_button.pack(pady=15)

result_label = tk.Label(root, text="Password Strength: ", font=("Helvetica", 18))
result_label.pack(pady=25)

root.mainloop()
