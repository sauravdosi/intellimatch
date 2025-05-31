import argparse
import logging
import pandas as pd
from kfold_tfidf_generator_driver import KFoldTFIDFGeneratorDriver
from nlp_preprocessing import NLPPreprocessing


# def setup_logging():
#     """
#     Configure logging settings for the workflow.
#     """
#     pass
#
#
# def parse_arguments():
#     """
#     Parse command-line arguments and return namespace.
#     """
#     parser = argparse.ArgumentParser(description="Controller for the TF-IDF preprocessing and generation workflow.")
#     parser.add_argument('--input', '-i', type=str, required=True,
#                         help='Path to input Excel file or JSON data source.')
#     parser.add_argument('--output', '-o', type=str, required=True,
#                         help='Path where the results will be saved.')
#     parser.add_argument('--n_process', '-n', type=int, default=None,
#                         help='Number of processes for TF-IDF generation.')
#     return parser.parse_args()


class WorkflowController:
    """
    Orchestrates the end-to-end workflow: loading, preprocessing, TF-IDF generation, and saving results.
    """

    def __init__(self, input_path: str, output_path: str, n_process: int = None):
        """
        Initialize controller with paths and processing options.

        Args:
            input_path: Path to the raw input data.
            output_path: Path for saving final results.
            n_process: Optional number of processes for parallelization.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.n_process = n_process
        self.pipeline = None
        self.df = pd.DataFrame()
        self.kfold_tfidf_generator_driver = KFoldTFIDFGeneratorDriver(input_path)
        self.nlp_preprocessing = NLPPreprocessing(self.df)

    def nlp_preprocessing_run(self):
        n_process = (len(self.df) // 500) + 1
        self.nlp_preprocessing = NLPPreprocessing(self.df, n_process=n_process)
        self.nlp_preprocessing.multiprocess_preprocess()
        result_df = self.nlp_preprocessing.result_df

        return result_df

    def initialize_pipeline(self):
        """
        Instantiate and configure the PreTFIDFPipeline.
        """
        pass

    def execute_pipeline(self):
        """
        Run the pipeline and return the resulting DataFrame.
        """
        self.df = self.kfold_tfidf_generator_driver.run()
        print("KFold TF-IDF generation complete.")
        print(self.df)

        df1 = self.df[((self.df["Source"] == "ISN Database") & (self.df["HasOwnerRole"] == 1))][:1000]
        df2 = self.df[(self.df["Source"] == "Salesforce")][:1000]

        self.df = pd.concat([df1, df2])

        self.df = self.nlp_preprocessing_run()
        print("NLP preprocessing complete.")
        print(self.df)

        return self.df

    def save_results(self, df):
        """
        Persist the DataFrame to the specified output path.

        Args:
            df: The DataFrame to save.
        """
        pass

    def run(self):
        """
        Full execution flow: setup, pipeline run, and saving.
        """
        # setup_logging()
        args = None  # placeholder for parsed args
        self.initialize_pipeline()
        result_df = self.execute_pipeline()
        self.save_results(result_df)


if __name__ == '__main__':
    # args = parse_arguments()
    controller = WorkflowController("data/pretfidf_prodtest.xlsx", "data/postnlppreprocessing.xlsx")
    controller.run()
