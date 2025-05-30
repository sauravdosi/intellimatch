import argparse
import os
import pandas as pd
from tfidf_generator import TFIDFGenerator
from tfidf_preprocessing import TFIDFPreprocessing
import multiprocessing as mp
import time
from configparser import ConfigParser


class KFoldTFIDFGenerator:
    def __init__(self, df, config_path="C:/Users/sdosi/ISN Summer 2024 Internship/Projects/"
                                       "Prospective Client Analysis/Python Outsourcing/config.ini"):
        self.config = ConfigParser()
        self.config.read([config_path])
        self.source_column = self.config.get("KFOLD_TFIDF_GENERATOR", "source_column")
        self.infer_source = self.config.get("KFOLD_TFIDF_GENERATOR", "infer_source")
        self.reference_source = self.config.get("KFOLD_TFIDF_GENERATOR", "reference_source")
        self.fold_size = int(self.config.get("KFOLD_TFIDF_GENERATOR", "fold_size"))

        self.tfidf_preprocessing = TFIDFPreprocessing(df, n_process=3)
        self.tfidf_generator = TFIDFGenerator(df, preprocess=False)
        self.tfidf_preprocessing.multiprocess_preprocess()
        self.df = self.tfidf_preprocessing.result_df
        self.sf_df = self.df[self.df[self.source_column] == self.infer_source]
        self.db_df = self.df[self.df[self.source_column] == self.reference_source]
        self.tfidf_generator.preprocessed_df = self.db_df
        self.tfidf_generator.compute_norm_tfidf_df()
        self.db_tfidf_df = self.tfidf_generator.tfidf_df

    @staticmethod
    def split_dataframe(df, n_process):
        chunk_size = len(df) // n_process
        return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    def run_multiprocess(self, n_process=4):
        # Splitting the input df into k-folds of 200 records each and
        # Concatenating every fold with the entire reference data from ISN Database tables
        df_chunks = self.split_dataframe(self.sf_df, n_process)

        df_chunks_combined = [pd.concat([df_chunk, self.db_df]) for df_chunk in df_chunks]

        print(len(df_chunks_combined))
        pool = mp.Pool(processes=n_process)
        result_chunks = pool.map(self.run, df_chunks_combined)

        pool.close()
        pool.join()

        result = [self.db_tfidf_df]
        result.extend(result_chunks)
        result_df = pd.concat(result)

        return result_df

    def run(self, df):
        partitions = self.k_fold_partition_df(df[df[self.source_column] == self.infer_source])

        k_fold_tfidf_dfs = []

        for partition_df in partitions:
            k_fold_df = pd.concat([self.db_df, partition_df], ignore_index=True)
            self.tfidf_generator.preprocessed_df = k_fold_df
            self.tfidf_generator.compute_norm_tfidf_df()
            k_fold_tfidf_df = self.tfidf_generator.tfidf_df
            k_fold_tfidf_dfs.append(k_fold_tfidf_df[k_fold_tfidf_df[self.source_column] == self.infer_source])

        print(len(self.db_tfidf_df))
        print(len(k_fold_tfidf_dfs))
        tfidf_df = [self.db_tfidf_df]
        tfidf_df.extend(k_fold_tfidf_dfs)
        print(len(tfidf_df))

        return pd.concat(k_fold_tfidf_dfs)

    def k_fold_partition_df(self, dataframe, partition_size=None):
        if partition_size is None:
            partition_size = self.fold_size

        k = (len(dataframe) + partition_size - 1) // partition_size

        df_shuffled = dataframe.sample(frac=1).reset_index(drop=True)

        partitions = [df_shuffled.iloc[i * partition_size:(i + 1) * partition_size] for i in range(k)]

        return partitions


class ArgParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Argument Parser for K-fold TFIDF Generator")
        self.add_arguments()
        self.args = None
        self.parse()

    def add_arguments(self):
        self.parser.add_argument("--i", "--input", type=str, required=True, help="Path to the input json file")
        self.parser.add_argument("--o", "--output", type=str, required=True, help="Path to the input json file")
        self.parser.add_argument("--np", "--n_process", type=int, default=4,
                                 help="Number of processes while multiprocessing")

    def parse(self):
        self.args = vars(self.parser.parse_args())

        if not os.path.isfile(self.args["i"]):
            self.parser.error(f"The input file {self.args["i"]} does not exist.")

        output_dir = os.path.dirname(self.args["o"])
        if output_dir and not os.path.isdir(output_dir):
            self.parser.error(f"The director {output_dir} does not exist.")

        if not self.args["o"].endswith(".json"):
            self.parser.error("The output file must be a JSON file.")


if __name__ == "__main__":
    start = time.time()
    mp.freeze_support()

    args = ArgParser().args
    df = pd.read_json(args["i"], orient="records")
    kfold_tfidf_generator = KFoldTFIDFGenerator(df)
    print(kfold_tfidf_generator.df)
    print(kfold_tfidf_generator.db_tfidf_df)

    result_df = kfold_tfidf_generator.run_multiprocess(args["np"])

    result_df = result_df[result_df["normalized_tfidf"] != {}]

    result_df.to_json(args["o"], orient="records")

    print(result_df)

    print(f"Time taken: {time.time() - start} seconds")
