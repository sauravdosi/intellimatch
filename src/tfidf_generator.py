import json
import pandas as pd
from collections import Counter
from src.tfidf_preprocessing import TFIDFPreprocessing


class TFIDFGenerator:
    def __init__(self, df, preprocess=True, idf_path=None):
        self.df = df
        if preprocess:
            self.tfidf_preprocessing = TFIDFPreprocessing(self.df)
            self.preprocessed_df = self.tfidf_preprocessing.result
        else:
            self.preprocessed_df = self.df
        self.tfidf_df = pd.DataFrame()
        self.idf_path = idf_path

    def compute_norm_tfidf_df(self):
        self.preprocessed_df["term_frequency"] = self.preprocessed_df["words"].apply(self.compute_tf)
        doc_frequency_counts = self.compute_df(self.preprocessed_df["words"])

        # Compute the inverse document frequency (IDF) using modified IDF function
        # The new function is fine-tuned to assign more importance to the rare words
        total_docs = len(self.preprocessed_df)
        idf = {term: (((total_docs / (1 + doc_frequency_counts[term])) ** 0.67) - 1) for term in doc_frequency_counts}

        if self.idf_path:
            with open(self.idf_path, "w") as file:
                json.dump(idf, file)

        self.tfidf_df = self.preprocessed_df
        self.tfidf_df["tfidf"] = self.tfidf_df["term_frequency"].apply(lambda x: self.compute_tfidf(x, idf))

        self.tfidf_df["normalized_tfidf"] = self.tfidf_df["tfidf"].apply(lambda x: self.normalize_tfidf(x))

    # Compute the term frequency (TF)
    @staticmethod
    def compute_tf(doc):
        tf = Counter(doc)
        doc_len = len(doc)
        for word in tf:
            tf[word] /= doc_len
        return tf

    # Compute the document frequency (DF)
    @staticmethod
    def compute_df(docs):
        df = Counter()
        for doc in docs:
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] += 1
        return df

    # Compute the TF-IDF
    @staticmethod
    def compute_tfidf(tf, idf):
        tfidf = {}
        for term in tf:
            tfidf[term] = tf[term] * idf.get(term)
        return tfidf

    @staticmethod
    def normalize_tfidf(tfidf_dict):
        total_score = sum(tfidf_dict.values())
        normalized_tfidf = {}
        for key, value in tfidf_dict.items():
            normalized_tfidf.update({key: value / total_score})
        return normalized_tfidf


