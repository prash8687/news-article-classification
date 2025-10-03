# Simple News Article Classification (placeholder demo)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Example data
texts = ["sports news about football", "politics news about election", "tech news about AI"]
labels = ["sports", "politics", "technology"]

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

# Test prediction
test = ["new AI breakthrough"]
X_test = vectorizer.transform(test)
print("Prediction:", model.predict(X_test)[0])
