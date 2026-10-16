# AI News Topic Classifier

A beginner machine-learning project that predicts a topic category for a short news headline.

## Workflow
1. Read labeled headline examples from `news_headlines.csv`.
2. Split data into training and testing sets.
3. Convert text to TF-IDF features.
4. Train a Logistic Regression classifier.
5. Print evaluation metrics and save the model pipeline.
6. Predict a category for a new headline.

## Run
From this folder:
```bash
python train.py
python predict.py
```

The bundled dataset is a small demonstration dataset, not a benchmark. Scores may vary and should not be presented as evidence of production-level performance.
