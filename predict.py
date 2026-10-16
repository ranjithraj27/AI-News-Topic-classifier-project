from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).with_name("news_classifier.joblib")

def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run `python train.py` first.")
        return

    model = joblib.load(MODEL_PATH)
    headline = input("Enter a news headline: ").strip()
    if not headline:
        print("Please enter a headline.")
        return

    prediction = model.predict([headline])[0]
    print(f"Predicted topic: {prediction}")

if __name__ == "__main__":
    main()
