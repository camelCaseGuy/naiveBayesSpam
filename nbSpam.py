# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
# You can download the SMS Spam Collection dataset from Kaggle or UCI Machine Learning Repository
url = 'https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/sms_spam.csv'
data = pd.read_csv(url, encoding='latin-1')

# Display the first few rows of the dataset
print(data.head())

# The dataset contains two columns: 'label' and 'message'
# 'label' contains the label (spam or ham), and 'message' contains the SMS text

# Data Preprocessing
# Let's clean the dataset by removing unnecessary columns and converting the text to lowercase
data = data[['label', 'message']]
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Convert labels to binary (ham: 0, spam: 1)

# Text cleaning
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def clean_text(text):
    text = text.lower()  # Lowercase the text
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = [word for word in text.split() if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(text)

data['cleaned_message'] = data['message'].apply(clean_text)

# Splitting the dataset into training and testing sets
X = data['cleaned_message']  # Features
y = data['label']  # Target (spam or ham)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=3000)  # Limit to the top 3000 features for efficiency
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Training the Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Predicting on the test set
y_pred = nb_model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Test the model with a custom input
def predict_spam(message):
    message_cleaned = clean_text(message)
    message_tfidf = tfidf.transform([message_cleaned])
    prediction = nb_model.predict(message_tfidf)
    return 'Spam' if prediction[0] == 1 else 'Ham'

# Example of custom input
custom_message = "Congratulations! You've won a $1,000 gift card. Call now to claim."
print(f"Message: {custom_message}")
print(f"Prediction: {predict_spam(custom_message)}")
