import pandas as pd
from src.keyword_classifier import KeywordClassifier

if __name__ == "__main__":
    df = pd.read_json("data/new_train_data.json", orient="records")
    keyword_classifier = KeywordClassifier(data=df, mode="train")
    keyword_classifier.run()