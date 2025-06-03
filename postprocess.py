import pandas as pd


class PostProcess:
    """
    Pipeline to process matched and unmatched records and merge with ISN active HCs,
    producing a final combined DataFrame.
    """
    def __init__(self, df: pd.DataFrame, active_hcs_df: pd.DataFrame):
        """
        Initialize the pipeline with main and active HCs DataFrames.

        Args:
            df: Main DataFrame containing match results. Must include columns:
                - "Match Score"
                - "Matched Company Name"
                - "AHC Company Name"
            active_hcs_df: DataFrame of ISN active historical company names (column "CompanyName").
        """
        self.df = df.copy()
        self.active_hcs_df = active_hcs_df.copy()
        self.unmatched_df = pd.DataFrame()
        self.matches_df = pd.DataFrame()
        self.merged_df = pd.DataFrame()
        self.final_df = pd.DataFrame()

    def split_matches(self):
        """
        Split the main df into unmatched and matched sets based on 'Match Score'.
        """
        self.unmatched_df = self.df[self.df["Match Score"] < 85]
        self.matches_df = self.df[self.df["Match Score"] >= 85]

    def merge_active_hcs(self):
        """
        Merge matched records with active HCs on company name.

        Produces self.merged_df with an 'ISN Database HC Match' column.
        """
        df = self.matches_df.copy()

        # Create a mapping of active historical names
        history_map = (
            self.active_hcs_df
            .groupby("CompanyName")["CompanyName"]
            .agg(list)
        )

        # Join history onto matched DataFrame
        df = df.join(history_map.rename("ISN Database HC Match"), on="Matched Company Name")

        # Fill missing lists
        df["ISN Database HC Match"] = df["ISN Database HC Match"].apply(
            lambda lst: lst if isinstance(lst, list) else []
        )

        self.merged_df = df

    def concatenate_and_compute(self) -> pd.DataFrame:
        """
        Combine matched (merged) records with unmatched,
        compute AHC name counts, deduplicate, and return final DataFrame.

        Returns:
            Final combined DataFrame.
        """
        # Combine
        unmatched = self.unmatched_df.copy()
        # Ensure the same columns exist
        unmatched["ISN Database HC Match"] = [[] for _ in range(len(unmatched))]
        unmatched["Matched Company Name"] = ""

        combined = pd.concat([self.merged_df, unmatched], ignore_index=True)

        # Compute uppercase counts for AHC Company Name
        combined["AHC Company Name Upper"] = combined["AHC Company Name"].apply(
            lambda x: x.upper().strip() if isinstance(x, str) else x
        )
        count_df = (
            combined["AHC Company Name Upper"]
            .value_counts()
            .reset_index()
        )
        count_df.columns = ["AHC Company Name Upper", "Count"]

        # Merge counts back
        combined = combined.merge(count_df, on="AHC Company Name Upper")

        # Drop duplicates on uppercase key (keeping first)
        final = combined.drop_duplicates(subset=["AHC Company Name Upper"]).copy()

        # Remove helper column
        final.drop(columns=["AHC Company Name Upper"], inplace=True)

        reference_columns = [
            "RecordID_CI", "AHC Company Name", "Source", "HasOwnerRole", "CompanyID",
            "CompanyNameOriginal", "isSeparated", "CompanyNameCleaned", "words", "term_frequency",
            "tfidf", "normalized_tfidf", "word_vecs", "ner_pos_dict", "ner", "pos", "pos_num",
            "pos_num_modified", "tfidf_values", "tfidf_minmax_normalized", "Predicted Labels",
            "Predicted Label Names", "label_list", "cleaned_words", "labels_dict",
            "Matched Company Name", "Matched Company Labels Dict", "Matched Company TFIDF",
            "Match Score", "Match Category", "Matched Company ISN ID",
            "Standardized Alias", "ISN Database HC Match", "Count"
        ]
        # Select only those present in the DataFrame, in that order
        ordered_columns = [col for col in reference_columns if col in final.columns]
        final = final[ordered_columns]

        return final

    def run(self) -> pd.DataFrame:
        """
        Execute the full workflow and return results.

        Returns:
            Final combined DataFrame after historical matching.
        """
        self.split_matches()
        self.merge_active_hcs()
        return self.concatenate_and_compute()

# Example usage:
# pipeline = HistoricalMatchingPipeline(main_df, active_hcs_df)
# final_df = pipeline.run()
