from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_PATH = Path(__file__).with_name("news_headlines.csv")
MODEL_PATH = Path(__file__).with_name("news_classifier.joblib")

def main():
    data = pd.read_csv(DATA_PATH)
    X = data["headline"].astype(str)
    y = data["category"].astype(str)

    # Stratification requires enough examples per category; the tiny demo data
    # is intentionally small, so use a simple split for this learning example.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Demo test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report (small sample; interpret cautiously):")
    print(classification_report(y_test, predictions, zero_division=0))

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
