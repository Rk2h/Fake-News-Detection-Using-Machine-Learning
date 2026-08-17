from flask import Flask, render_template, request, send_file
import csv
import joblib
import json
import os
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from preprocessing import clean_text

app = Flask(__name__)

# ==========================
# Load Model Files
# ==========================

model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
label_map = joblib.load("model/label_map.pkl")
model_accuracy = joblib.load("model/model_accuracy.pkl")

# ==========================
# History / Downloads
# ==========================

HISTORY_FILE = "history.json"
DOWNLOADS_DIR = "downloads"

os.makedirs(DOWNLOADS_DIR, exist_ok=True)


def load_history():
    """Load prediction history from history.json."""
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_history(history):
    """Save prediction history to history.json."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4, ensure_ascii=False)


def get_analytics(history):
    """Calculate dashboard statistics and chart data."""
    total_predictions = len(history)

    real_count = sum(
        1
        for item in history
        if item.get("prediction") == "REAL NEWS"
    )

    fake_count = sum(
        1
        for item in history
        if item.get("prediction") == "FAKE NEWS"
    )

    if history:
        confidences = [
            float(item.get("confidence", 0))
            for item in history
        ]

        average_confidence = round(
            sum(confidences) / len(confidences),
            2
        )

        highest_confidence = max(confidences)
        lowest_confidence = min(confidences)

    else:
        average_confidence = 0
        highest_confidence = 0
        lowest_confidence = 0

    # Oldest -> newest for the trend chart
    confidence_history = []
    prediction_labels = []

    for i, item in enumerate(reversed(history), start=1):
        confidence_history.append(
            float(item.get("confidence", 0))
        )
        prediction_labels.append(f"P{i}")

    return {
        "total_predictions": total_predictions,
        "real_count": real_count,
        "fake_count": fake_count,
        "average_confidence": average_confidence,
        "highest_confidence": highest_confidence,
        "lowest_confidence": lowest_confidence,
        "confidence_history": confidence_history,
        "prediction_labels": prediction_labels,
    }


# ==========================
# Home Page
# ==========================

@app.route("/")
def home():
    history = load_history()
    analytics = get_analytics(history)

    return render_template(
        "index.html",
        prediction=None,
        is_real=None,
        confidence=None,
        confidence_level=None,
        confidence_color=None,
        accuracy=round(model_accuracy * 100, 2),
        news="",
        history=history,
        top_keywords=[],
        xai_features=[],
        xai_supporting=[],
        xai_against=[],
        xai_summary=None,
        insight=None,
        recommendation=None,
        **analytics
    )


# ==========================
# Prediction
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form.get("news", "").strip()

    # ==========================
    # Validate Input
    # ==========================

    if not news:
        history = load_history()

        return render_template(
            "index.html",
            prediction=None,
            is_real=None,
            confidence=None,
            confidence_level=None,
            confidence_color=None,
            accuracy=round(model_accuracy * 100, 2),
            news="",
            history=history,
            top_keywords=[],
            xai_features=[],
            xai_supporting=[],
            xai_against=[],
            xai_summary=None,
            insight=None,
            recommendation=None,
            error="Please enter a news article before analyzing.",
            **get_analytics(history)
        )

    # ==========================
    # Clean Text
    # ==========================

    clean_news = clean_text(news)

    # ==========================
    # TF-IDF Vector
    # ==========================

    news_vector = vectorizer.transform([clean_news])

    # ==========================
    # Prediction
    # ==========================

    prediction = model.predict(news_vector)[0]
    result = str(label_map[prediction]).strip().upper()



    # Boolean used by HTML
    is_real = result == "REAL NEWS"

    # ==========================
    # Confidence
    # ==========================

    probabilities = model.predict_proba(news_vector)[0]
    confidence = round(float(max(probabilities)) * 100, 2)


    # ==========================
    # Top Important Keywords
    # ==========================

    feature_names = vectorizer.get_feature_names_out()
    vector_array = news_vector.toarray()[0]

    indices = vector_array.argsort()[-10:][::-1]

    top_keywords = []

    for index in indices:
        if vector_array[index] > 0:
            top_keywords.append(feature_names[index])

    # ==========================
    # Explainable AI (XAI)
    # "Why did the AI make this prediction?"
    # ==========================

    xai_features = []

    try:
        # Logistic Regression coefficients
        coefficients = model.coef_[0]

        # XAI below is designed for binary Logistic Regression.
        # If a different model is loaded, leave the app functional
        # and simply show no XAI features.
        if len(model.classes_) != 2:
            raise ValueError("XAI explanation currently supports binary classification only.")

        # Contribution of every TF-IDF feature
        contributions = vector_array * coefficients

        # For binary Logistic Regression, a positive contribution
        # moves the decision toward model.classes_[1], while a
        # negative contribution moves it toward model.classes_[0].
        # Reverse the sign when class 0 is the predicted class so that
        # positive contribution always means "supports the prediction".
        if prediction == model.classes_[1]:
            feature_contributions = contributions
        else:
            feature_contributions = -contributions

        # Only words that actually appear in this article
        active_indices = [
            i
            for i, value in enumerate(vector_array)
            if value > 0
        ]

        # Sort by absolute influence
        active_indices = sorted(
            active_indices,
            key=lambda i: abs(feature_contributions[i]),
            reverse=True
        )

        # Top 8 words influencing the prediction
        for index in active_indices[:8]:

            contribution = float(feature_contributions[index])

            xai_features.append({
                "word": feature_names[index],
                # Keep the signed contribution so the explanation
                # represents the actual direction of the model.
                "score": round(abs(contribution), 4),

                # Used by the HTML template to decide the badge/color.
                # "support" = pushes toward the predicted class.
                # "against" = pushes away from the predicted class.
                "support": (
                    "support"
                    if contribution > 0
                    else "against"
                ),

                # Kept for compatibility if another part of the UI
                # wants to use the older field name.
                "direction": (
                    "support"
                    if contribution > 0
                    else "against"
                )
            })

    except (AttributeError, IndexError, ValueError, TypeError):
        # Keep the application working even if XAI
        # is not available for the loaded model.
        xai_features = []

    # ==========================
    # XAI Visualization Groups
    # ==========================
    # These are only presentation helpers.
    # The trained model and prediction are NOT changed.

    xai_supporting = [
        item for item in xai_features
        if item.get("support") == "support"
    ]

    xai_against = [
        item for item in xai_features
        if item.get("support") == "against"
    ]

    # Strongest factors first
    xai_supporting.sort(
        key=lambda item: float(item.get("score", 0)),
        reverse=True
    )

    xai_against.sort(
        key=lambda item: float(item.get("score", 0)),
        reverse=True
    )

    # Use the strongest XAI feature as 100% for the visual bars.
    # Influence scores themselves remain unchanged.
    all_xai_scores = [
        float(item.get("score", 0))
        for item in xai_features
    ]

    max_xai_score = max(all_xai_scores, default=0)

    for item in xai_features:
        score = float(item.get("score", 0))

        if max_xai_score > 0:
            item["impact_percent"] = round(
                (score / max_xai_score) * 100, 1
            )
        else:
            item["impact_percent"] = 0

    # Rebuild grouped lists after adding visualization data.
    xai_supporting = [
        item for item in xai_features
        if item.get("support") == "support"
    ]

    xai_against = [
        item for item in xai_features
        if item.get("support") == "against"
    ]

    xai_supporting.sort(
        key=lambda item: float(item.get("score", 0)),
        reverse=True
    )

    xai_against.sort(
        key=lambda item: float(item.get("score", 0)),
        reverse=True
    )

    if xai_features:
        strongest_xai = max(
            xai_features,
            key=lambda item: float(item.get("score", 0))
        )

        xai_summary = (
            f"The strongest identified feature is "
            f"'{strongest_xai.get('word', '')}', with an influence "
            f"score of {float(strongest_xai.get('score', 0)):.4f}. "
        )

        if strongest_xai.get("support") == "support":
            xai_summary += (
                "It supports the model's final prediction."
            )
        else:
            xai_summary += (
                "It pushes the model away from the final prediction."
            )
    else:
        xai_summary = (
            "The model did not provide enough active features "
            "to generate an explanation."
        )

    # ==========================
    # Confidence Level
    # ==========================

    if confidence >= 90:
        confidence_level = "Very High Confidence"
        confidence_color = "success"

    elif confidence >= 75:
        confidence_level = "High Confidence"
        confidence_color = "primary"

    elif confidence >= 60:
        confidence_level = "Moderate Confidence"
        confidence_color = "warning"

    else:
        confidence_level = "Low Confidence"
        confidence_color = "danger"

    # ==========================
    # AI Prediction Insights
    # ==========================

    if is_real:
        insight = (
            "The AI model predicts this article is REAL NEWS. "
            "The content matches patterns commonly found in reliable news."
        )

    else:
        insight = (
            "The AI model predicts this article is FAKE NEWS. "
            "The content contains patterns frequently associated "
            "with misinformation."
        )

    # ==========================
    # Recommendation
    # ==========================

    if confidence >= 90:
        recommendation = "Prediction is highly reliable."

    elif confidence >= 75:
        recommendation = "Prediction is reliable."

    elif confidence >= 60:
        recommendation = "Prediction is moderately reliable."

    else:
        recommendation = "Prediction should be reviewed manually."

    # ==========================
    # Save Prediction History
    # ==========================

    history = load_history()

    history.insert(
        0,
        {
            "news": news[:120] + ("..." if len(news) > 120 else ""),
            "prediction": result,
            "confidence": confidence,
            "datetime": datetime.now().strftime(
                "%d-%m-%Y %H:%M:%S"
            ),
        },
    )

    # Keep latest 10 predictions
    history = history[:10]

    save_history(history)

    # ==========================
    # Analytics
    # ==========================

    analytics = get_analytics(history)

    # ==========================
    # Render Template
    # ==========================

    return render_template(
        "index.html",

        prediction=result,
        is_real=is_real,

        confidence=confidence,
        confidence_level=confidence_level,
        confidence_color=confidence_color,

        accuracy=round(model_accuracy * 100, 2),

        news=news,
        history=history,

        top_keywords=top_keywords,

        # Explainable AI
        xai_features=xai_features,
        xai_supporting=xai_supporting,
        xai_against=xai_against,
        xai_summary=xai_summary,

        insight=insight,
        recommendation=recommendation,

        **analytics
    )


# ==========================
# Download PDF Report
# ==========================

@app.route("/download_report")
def download_report():

    history = load_history()

    if not history:
        return "No prediction available."

    latest = history[0]

    filename = os.path.join(
        DOWNLOADS_DIR,
        "Fake_News_Report.pdf"
    )

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()
    story = []

    story.append(
        Paragraph(
            "<b>Fake News Detection AI</b>",
            styles["Title"]
        )
    )

    story.append(
        Paragraph(
            "<br/>",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            "<b>Prediction:</b> "
            + str(latest.get("prediction", "")),
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            "<b>Confidence:</b> "
            + str(latest.get("confidence", 0))
            + "%",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            "<b>Date &amp; Time:</b> "
            + str(latest.get("datetime", "")),
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            "<br/>",
            styles["Normal"]
        )
    )

    news_text = (
        str(latest.get("news", ""))
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )

    story.append(
        Paragraph(
            "<b>News:</b><br/>" + news_text,
            styles["BodyText"]
        )
    )

    doc.build(story)

    return send_file(
        filename,
        as_attachment=True
    )


# ==========================
# Export CSV
# ==========================

@app.route("/export_csv")
def export_csv():

    history = load_history()

    filename = os.path.join(
        DOWNLOADS_DIR,
        "prediction_history.csv"
    )

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "News",
            "Prediction",
            "Confidence (%)",
            "Date & Time"
        ])

        for item in history:

            writer.writerow([
                item.get("news", ""),
                item.get("prediction", ""),
                item.get("confidence", ""),
                item.get("datetime", "")
            ])

    return send_file(
        filename,
        as_attachment=True
    )


# ==========================
# Run
# ==========================

if __name__ == "__main__":
    app.run(debug=True)