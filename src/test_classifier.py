import pandas as pd
from src.keyword_classifier import KeywordClassifier

if __name__ == "__main__":
    df = pd.read_json("data/ml_matching_train_data5.json", orient="records")
    keyword_classifier = KeywordClassifier(data=df, mode="test", model="keyword_classifier.weights.h5")
    keyword_classifier.run()
    print(keyword_classifier.data_pred)
    keyword_classifier.data_pred.to_csv("data/kw_test.csv", index=False)