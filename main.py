# AI-based Phishing Email Detection with Advanced Features

# Install necessary libraries
# !pip install pandas scikit-learn numpy nltk

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Sample Dataset Generation
data = {
    'text': [
        'Urgent! Your account needs verification. Click here to update.',
        'You won a lottery! Claim your prize now!',
        'Please find the project report attached.',
        'Verify your bank details to avoid suspension.',
        'Meeting scheduled for 2 PM tomorrow.',
        'Limited time offer! Get free gift cards now!',
        'Final reminder: Update your payment info to avoid late fees.'
    ],
    'label': ['phishing', 'phishing', 'legitimate', 'phishing', 'legitimate', 'phishing', 'phishing']
}

# Create DataFrame
dataset = pd.DataFrame(data)

# Feature Engineering
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special chars
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

dataset['clean_text'] = dataset['text'].apply(preprocess_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(dataset['clean_text']).toarray()
y = dataset['label']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Explanation:
# - Preprocessing removes URLs, special characters, and stopwords.
# - TF-IDF vectorizer extracts numerical features from text.
# - Random Forest is used for classification due to its robustness.
# - Accuracy and classification report are displayed after training.
