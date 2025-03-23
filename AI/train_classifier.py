import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle


training_data = pd.read_csv("AI/training_data.csv")
X = training_data["text"]
y = training_data["label"]


# Vectorize the text
vectorizer = TfidfVectorizer(max_features=100)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec,y , test_size=0.2, random_state=42)
clf = LogisticRegression()
clf.fit(X_train, y_train)

print("Accuracy:", clf.score(X_test, y_test))

with open("AI/classifier.pkl", "wb") as f:
    pickle.dump({'classifier': clf, 'vectorizer': vectorizer}, f)