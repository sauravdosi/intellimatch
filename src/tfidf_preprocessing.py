import re
import multiprocessing as mp
import pandas as pd
from configparser import ConfigParser


class TFIDFPreprocessing:
    def __init__(self, df, n_process=1, config_path="config/config.ini"):
        self.config = ConfigParser()
        self.config.read([config_path])
        self.input_column = self.config.get("TFIDF_PREPROCESSING", "input_column")
        self.output_column = self.config.get("TFIDF_PREPROCESSING", "output_column")

        self.df = df
        self.n_process = n_process
        self.result_df = pd.DataFrame()

    def split_dataframe(self):
        chunk_size = len(self.df) // self.n_process
        return [self.df[i:i + chunk_size] for i in range(0, len(self.df), chunk_size)]

    def multiprocess_preprocess(self):
        mp.freeze_support()
        df_chunks = self.split_dataframe()
        pool = mp.Pool(processes=self.n_process)
        result_chunks = pool.map(self.preprocess, df_chunks)
        pool.close()
        pool.join()
        result_df = pd.concat(result_chunks)
        self.result_df = result_df

    def preprocess(self, df):
        df = self.upper_case(df)
        df = self.tokenize(df)
        df = self.clean(df)
        return df

    def upper_case(self, df):
        # Convert the Company Names to upper case
        df["CompanyNameCleaned"] = df[self.input_column].apply(lambda x: x.upper() if type(x) == str else "")
        # df["CompanyNameCleaned"] = df["CompanyNameCleaned"].apply(lambda x: re.sub(r"\bTHE\b", "", x) if type(x) == str else "")
        return df

    def tokenize(self, df):
        # Tokenize the text data
        df[self.output_column] = df["CompanyNameCleaned"].apply(lambda x: x.split(" "))
        return df

    def clean(self, df):
        # Replace & (special character) with AND
        df[self.output_column] = df[self.output_column].apply(lambda l: list(map(lambda x: x.replace('&', 'AND') if x == "&" else x, l)))

        # Filter special characters from the words
        df[self.output_column] = df[self.output_column].apply(lambda x: [re.sub(r"[^A-Za-z0-9\&]", "", i) for i in x])

        # Filter single letter words: Might need to check
        # df["words"] = df["words"].apply(lambda x: [word for word in x if len(word) > 1])

        # Remove empty strings
        df[self.output_column] = df[self.output_column].apply(lambda x: [word for word in x if word])

        return df

