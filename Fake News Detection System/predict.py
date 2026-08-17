import joblib
from preprocessing import clean_text

# Load model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# User Input
news = input("Enter News:\n")

# Clean the news
clean_news = clean_text(news)

# Convert to TF-IDF
news_vector = vectorizer.transform([clean_news])

# Predict
prediction = model.predict(news_vector)

print("\n" + "=" * 40)

if prediction[0] == 0:
    print("Prediction : FAKE NEWS")
else:
    print("Prediction : REAL NEWS")

print("=" * 40)