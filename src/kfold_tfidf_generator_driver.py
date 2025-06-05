import time
import re
import multiprocessing as mp
from typing import List, Union, Optional
from configparser import ConfigParser
import pandas as pd
from src.kfold_tfidf_generator import KFoldTFIDFGenerator


class KFoldTFIDFGeneratorDriver:
    """
    Pipeline to preprocess company names and generate K-Fold TF-IDF features.

    Attributes:
        input_source (Union[str, pd.DataFrame]): Path to the Excel file or preloaded DataFrame.
        df (pd.DataFrame): DataFrame after initial load.
        n_process (Optional[int]): Number of processes to use for TF-IDF generation; if None, calculated dynamically.
    """

    def __init__(self,
                 input_source: Union[str, pd.DataFrame],
                 n_process: Optional[int] = None,
                 config="config/config.ini") -> None:
        """
        Initialize the pipeline with a data source and optional number of processes.

        Args:
            input_source: File path to Excel or a pandas DataFrame.
            n_process: Number of parallel processes for TF-IDF generation. If None, it will be calculated.
        """
        self.input_source = input_source
        self.n_process = n_process
        self.df: pd.DataFrame = pd.DataFrame()
        self.config = ConfigParser()
        self.config.read(config)
        self.reference_column = self.config.get("KFOLD_TFIDF_GENERATOR", "reference_source")

    def load_data(self) -> pd.DataFrame:
        """
        Load the data from an Excel file or use the provided DataFrame.
        """
        if isinstance(self.input_source, str):
            self.df = pd.read_excel(self.input_source)
        elif isinstance(self.input_source, pd.DataFrame):
            self.df = self.input_source.copy()
        else:
            raise ValueError("Unsupported input source type.")
        return self.df

    @staticmethod
    def _split_on_slash(text: str) -> List[str]:
        """
        Split a company name on slashes, removing 'c/o' or 'contractor/operator'.

        Args:
            text: The input string to split.

        Returns:
            A list of cleaned name parts.
        """
        name = re.sub(r"\b(c\s*/\s*o|contractor\s*/\s*operator)\b", "", text, flags=re.IGNORECASE)
        parts = [part.strip() for part in name.split("/")]
        return [part for part in parts if part]

    def _separate_company_names(self, record: pd.Series) -> List[str]:
        """
        Extract and separate company names based on custom rules.

        Args:
            record: A pandas Series representing a row in the DataFrame.

        Returns:
            List of extracted company names.
        """
        text = record["CompanyName"]
        companies = [text]

        if record.get("Source") == self.reference_column:
            companies = []
            if isinstance(text, str):
                text = text.strip()
                if text.endswith(")") and "(" in text:
                    open_idx = text.index("(")
                    base = text[:open_idx].strip()
                    bracket = text[open_idx + 1:-1]
                    companies.append(base)
                    if "/" in bracket:
                        companies.extend(self._split_on_slash(bracket))
                    elif not any(x in bracket for x in ["APAC", "EMEA"]) and not re.match(r"^\d{3}-\d{6}$", bracket):
                        companies.append(bracket)
                    else:
                        companies = [text]
                elif "/" in text:
                    companies.extend(self._split_on_slash(text))
                else:
                    companies.append(text)
                companies = [c.strip() for c in companies if isinstance(c, str) and c]

        return companies

    def process_data(self) -> pd.DataFrame:
        """
        Apply company name separation and explode the DataFrame.

        Returns:
            The processed DataFrame.
        """
        self.df["CompanyNameOriginal"] = self.df["CompanyName"]
        self.df["CompanyName"] = self.df.apply(self._separate_company_names, axis=1)
        self.df["isSeparated"] = self.df["CompanyName"].apply(lambda lst: len(lst) > 1)
        self.df = self.df.explode("CompanyName").reset_index(drop=True)
        return self.df

    def run(self) -> pd.DataFrame:
        """
        Execute the full preprocessing and K-Fold TF-IDF generation.

        Returns:
            DataFrame containing the normalized TF-IDF results.
        """
        start_time = time.time()
        mp.freeze_support()

        # Load and preprocess data
        self.load_data()
        processed_df = self.process_data()

        # Determine number of processes if not provided
        num_process = self.n_process if self.n_process is not None else (
            (len(processed_df[processed_df["Source"] == "Salesforce"]) // 10000) + 1
        )

        # Generate K-Fold TF-IDF
        kfold_generator = KFoldTFIDFGenerator(processed_df)
        result = kfold_generator.run_multiprocess(num_process)
        result = result[result["normalized_tfidf"] != {}]

        print(f"Pipeline completed in {time.time() - start_time:.2f} seconds using {num_process} processes.")
        return result


if __name__ == "__main__":
    pipeline = KFoldTFIDFGeneratorDriver("data/pretfidf_prodtest.xlsx")
    output_df = pipeline.run()
    print(output_df)

