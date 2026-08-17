import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing import clean_text


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

print("Fake News Articles :", len(fake))
print("Real News Articles :", len(true))


# ============================================================
# 2. ADD LABELS
# ============================================================

# 0 = Fake News
# 1 = Real News

fake["label"] = 0
true["label"] = 1


# ============================================================
# 3. COMBINE DATASETS
# ============================================================

data = pd.concat(
    [fake, true],
    ignore_index=True
)

# Shuffle dataset
data = data.sample(
    frac=1,
    random_state=42
)

data.reset_index(
    drop=True,
    inplace=True
)

print("\nCombined Dataset Shape:", data.shape)


# ============================================================
# 4. COMBINE TITLE + TEXT
# ============================================================

data["content"] = (
    data["title"].fillna("") +
    " " +
    data["text"].fillna("")
)


# ============================================================
# 5. CLEAN TEXT
# ============================================================

print("\n" + "=" * 70)
print("TEXT PREPROCESSING")
print("=" * 70)

data["content"] = data["content"].apply(clean_text)

print("Text preprocessing completed.")


# ============================================================
# 6. FEATURES AND LABELS
# ============================================================

X = data["content"]
y = data["label"]


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples :", len(X_train_text))
print("Testing Samples  :", len(X_test_text))


# ============================================================
# 8. TF-IDF VECTORIZATION
# ============================================================

print("\n" + "=" * 70)
print("TF-IDF VECTORIZATION")
print("=" * 70)

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

# Fit ONLY on training data
X_train = vectorizer.fit_transform(
    X_train_text
)

# Transform test data using same vectorizer
X_test = vectorizer.transform(
    X_test_text
)

print("Training Feature Matrix:", X_train.shape)
print("Testing Feature Matrix :", X_test.shape)


# ============================================================
# 9. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ============================================================
# 10. PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

# Training predictions
y_train_pred = model.predict(X_train)

# Testing predictions
y_test_pred = model.predict(X_test)

print("Predictions generated successfully.")


# ============================================================
# 11. TRAINING ACCURACY
# ============================================================

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)


# ============================================================
# 12. TESTING ACCURACY
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)


# ============================================================
# 13. PRECISION / RECALL / F1
# ============================================================

precision = precision_score(
    y_test,
    y_test_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_test_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_test_pred,
    average="weighted"
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred
)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_test_pred,
    target_names=[
        "FAKE NEWS",
        "REAL NEWS"
    ],
    digits=4
)


# ============================================================
# 16. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION RESULTS")
print("=" * 70)

print(
    f"\nTraining Accuracy : "
    f"{train_accuracy * 100:.2f}%"
)

print(
    f"Testing Accuracy  : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Precision         : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall            : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1-Score          : "
    f"{f1 * 100:.2f}%"
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("\n                 Predicted")
print("              Fake     Real")
print(
    f"Actual Fake   {cm[0][0]:5d}    {cm[0][1]:5d}"
)
print(
    f"Actual Real   {cm[1][0]:5d}    {cm[1][1]:5d}"
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# 19. SAVE EVALUATION METRICS
# ============================================================

metrics = {
    "training_accuracy": train_accuracy,
    "testing_accuracy": test_accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "confusion_matrix": cm.tolist()
}

joblib.dump(
    metrics,
    "model/evaluation_metrics.pkl"
)


# ============================================================
# 20. SAVE MODEL ACCURACY
# ============================================================

# Your existing Flask app uses this file.

joblib.dump(
    test_accuracy,
    "model/model_accuracy.pkl"
)


# ============================================================
# 21. SAVE LABEL MAP
# ============================================================

label_map = {
    0: "FAKE NEWS",
    1: "REAL NEWS"
}

joblib.dump(
    label_map,
    "model/label_map.pkl"
)


# ============================================================
# 22. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    "model/fake_news_model.pkl"
)


# ============================================================
# 23. SAVE TF-IDF VECTORIZER
# ============================================================

joblib.dump(
    vectorizer,
    "model/tfidf_vectorizer.pkl"
)


# ============================================================
# 24. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 70)

print("\nSaved files:")

print("✓ model/fake_news_model.pkl")
print("✓ model/tfidf_vectorizer.pkl")
print("✓ model/label_map.pkl")
print("✓ model/model_accuracy.pkl")
print("✓ model/evaluation_metrics.pkl")

print("\nTraining and evaluation completed successfully!")