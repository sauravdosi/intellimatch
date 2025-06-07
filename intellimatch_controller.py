import pandas as pd
import multiprocessing as mp
from configparser import ConfigParser
from src.keyword_classifier import KeywordClassifier
from src.kfold_tfidf_generator_driver import KFoldTFIDFGeneratorDriver
from src.nlp_preprocessing import NLPPreprocessing
from src.ml_fuzzy_matching import MLFuzzyMatching
from src.postprocess import PostProcess


class IntelliMatchController:
    """
    Orchestrates the end-to-end workflow: loading, preprocessing, TF-IDF generation, and saving results.
    """

    def __init__(self, input_path: str, output_path: str, n_process: int = None, config="config/config.ini"):
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
        self.keyword_classifier = KeywordClassifier(self.df)
        self.ml_fuzzy_matching = MLFuzzyMatching(self.df)
        self.postprocess = PostProcess(self.df, self.df)
        self.config = ConfigParser()
        self.config.read(config)
        self.reference_column = self.config.get("KFOLD_TFIDF_GENERATOR", "reference_source")
        self.inference_column = self.config.get("KFOLD_TFIDF_GENERATOR", "infer_source")

    def nlp_preprocessing_run(self):
        n_process = (len(self.df) // 50000) + 1
        n_process = 4
        print("Running NLP Preprocessing...")
        self.nlp_preprocessing = NLPPreprocessing(self.df, n_process=n_process)
        self.nlp_preprocessing.multiprocess_preprocess()
        result_df = self.nlp_preprocessing.result_df
        print("Controller NLP Preprocessing done.")

        return result_df

    def keyword_classifier_run(self):
        mode = "test"
        model = "ml_matching9_model.weights.h5"
        self.keyword_classifier = KeywordClassifier(data=self.df, mode=mode, model=model)
        self.keyword_classifier.run()
        result_df = self.keyword_classifier.data_pred

        return result_df

    def ml_fuzzy_matching_run(self):
        n_process = 4
        self.ml_fuzzy_matching = MLFuzzyMatching(self.df)
        print(self.ml_fuzzy_matching.reference_df)

        # Reference data is first only subscribed HCs
        # k + l (all infer df) -> m (only subscribed HCs)
        self.ml_fuzzy_matching.reference_df = self.df[(self.df["Source"] == self.reference_column) & (self.df["HasOwnerRole"] == 1)]
                                                      # & (self.df["Database Table"] != "DMSalesforce.dim.SalesforceAccount")]

        print(self.ml_fuzzy_matching.infer_df)
        print(self.ml_fuzzy_matching.reference_df)

        result_df = self.ml_fuzzy_matching.match_multiprocess(n_process)

        self.ml_fuzzy_matching.refer_company_column = "AHC Company Name"
        self.ml_fuzzy_matching.matched_company_name = "Standardized Alias"
        self.ml_fuzzy_matching.infer_df = result_df
        self.ml_fuzzy_matching.reference_df = result_df
        self.ml_fuzzy_matching.self_match = True

        final_result_df = self.ml_fuzzy_matching.match_multiprocess(n_process)

        return final_result_df

    def postprocess_run(self):
        self.postprocess = PostProcess(self.df, pd.read_excel("data/isn_active_hcs.xlsx"))
        return self.postprocess.run()

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
        # self.df.to_csv("data/kfold_tfidf.csv", index=False)

        df1 = self.df[((self.df["Source"] == self.reference_column) & (self.df["HasOwnerRole"] == 1))][:]
        df2 = self.df[(self.df["Source"] == "Salesforce")][:100]

        self.df = pd.concat([df1, df2])

        self.df = self.nlp_preprocessing_run()
        print("NLP preprocessing complete.")
        print(self.df)
        self.df.to_csv("data/nlp_preprocessing.csv", index=False)

        self.df = self.keyword_classifier_run()
        print("Keyword classifier complete.")
        print(self.df)
        self.df.to_csv("data/keyword_classifier.csv", index=False)

        self.df = self.ml_fuzzy_matching_run()
        print("ML fuzzy matching complete.")
        print(self.df)
        self.df.to_csv("data/ml_fuzzy_matching.csv", index=False)

        self.df = self.postprocess_run()
        print("Postprocessing complete.")
        print(self.df)
        self.df.to_csv("data/postprocess.csv", index=False)

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
        self.initialize_pipeline()
        result_df = self.execute_pipeline()
        self.save_results(result_df)


if __name__ == '__main__':
    mp.freeze_support()
    controller = IntelliMatchController("data/pretfidf_prodtest.xlsx", "data/postnlppreprocessing.xlsx")
    controller.run()
