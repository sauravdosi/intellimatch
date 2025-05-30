import argparse
import time
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
import ast
import multiprocessing as mp
from functools import partial
from pyphonetics import Metaphone
from kfold_tfidf_generator import ArgParser as KFoldTFIDFArgParser


class TFIDFFuzzyMatching:
    def __init__(self, df, refer_company_column="CompanyName", matched_company_name="Matched Company Name",
                 self_match=False):
        self.infer_df = df[df["Source"] == "Salesforce"]
        self.reference_df = df[(df["Source"] == "ISN Database") & (df["HasOwnerRole"] == 1)]
        print(len(self.reference_df))
        if "CompanyName" in self.infer_df.columns.tolist():
            self.infer_df = self.infer_df.rename(columns={"CompanyName": "AHC Company Name"})
        # Keyword weights for the comparison and contribution of each keyword class towards the match score
        self.keyword_weights = {"0": 0.85, "1": 0.1, "2": 0.05}
        self.metaphone = Metaphone()
        self.refer_company_column = refer_company_column
        self.matched_company_name = matched_company_name
        self.self_match = self_match

    def match_multiprocess(self, n_process=4):
        df_chunks = self.split_dataframe(self.infer_df, n_process)

        print(len(df_chunks))
        pool = mp.Pool(processes=n_process)
        if self.self_match:
            result_chunks = pool.map(self.match_self, df_chunks)
        else:
            result_chunks = pool.map(self.match, df_chunks)

        pool.close()
        pool.join()

        result_df = pd.concat(result_chunks)

        return result_df

    @staticmethod
    def split_dataframe(df, n_process):
        chunk_size = len(df) // n_process
        if chunk_size == 0:
            return [df]
        else:
            return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    def match_self(self, infer_df):
        if not infer_df.empty:
            print(len(infer_df))
            infer_df["fuzzy_inference"] = infer_df.apply(
                lambda x: self.infer_fuzz_self(x["labels_dict"], self.reference_df) if x["cleaned_words"]
                else {}, axis=1)

            infer_df = infer_df.explode("fuzzy_inference")
            infer_df["Standardized Alias"] = infer_df["fuzzy_inference"].apply(
                lambda x: x[self.refer_company_column] if type(x) == dict and x else "")

            infer_df = infer_df.drop(columns=["fuzzy_inference"])

        else:
            infer_df[self.matched_company_name] = ""
            infer_df["Match Type"] = ""

        return infer_df

    def infer_fuzz_self(self, labels_dict2, refer_df):
        reference_df_copy = refer_df.copy()
        reference_df_copy[["fuzzy_score", "match_type"]] = reference_df_copy.apply(
            lambda x: self.calculate_fuzzy_score_new(x["labels_dict"], labels_dict2), axis=1, result_type="expand")

        sort_df = reference_df_copy[(reference_df_copy["match_type"] == "Match")
                                    | (reference_df_copy["match_type"] == "Alias")][[
            self.refer_company_column, "match_type"]]
        print(labels_dict2)
        print(sort_df)
        sort_df = sort_df.sort_values(
            by=self.refer_company_column, key=lambda x: x.str.len(), ascending=False).reset_index(drop=True).head(1)
        return sort_df.to_dict(orient="records") if not sort_df.empty else {}

    def match(self, infer_df):
        if not infer_df.empty:
            print(len(infer_df))
            infer_df["fuzzy_inference"] = infer_df.apply(
                lambda x: self.infer_fuzz(x["labels_dict"], self.reference_df) if x["cleaned_words"]
                else {}, axis=1)

            infer_df = infer_df.explode("fuzzy_inference")
            infer_df[self.matched_company_name] = infer_df["fuzzy_inference"].apply(
                lambda x: x[self.refer_company_column] if type(x) == dict and x else "")
            infer_df["Matched Company Labels Dict"] = infer_df["fuzzy_inference"].apply(
                lambda x: x["labels_dict"] if type(x) == dict and x else "")
            infer_df["Matched Company TFIDF"] = infer_df["fuzzy_inference"].apply(
                lambda x: x["normalized_tfidf"] if type(x) == dict and x else "")
            infer_df["Match Score"] = infer_df["fuzzy_inference"].apply(
                lambda x: x["fuzzy_score"] if type(x) == dict and x else "")
            infer_df["Match Category"] = infer_df["fuzzy_inference"].apply(
                lambda x: x["match_type"] if type(x) == dict and x else "")
            infer_df["Matched Company ID_CI"] = infer_df["fuzzy_inference"].apply(
                lambda x: x["ID_CI"] if type(x) == dict and x else "")
            infer_df["Matched Company ISN ID"] = infer_df["fuzzy_inference"].apply(
                lambda x: x["CompanyID"] if type(x) == dict and x else "")
            infer_df["Matched from Database Table"] = infer_df["fuzzy_inference"].apply(
                lambda x: x["Database Table"] if type(x) == dict and x else "")
            infer_df["Matched Demo Prospect Client"] = infer_df["fuzzy_inference"].apply(
                lambda x: x["isDemo"] if type(x) == dict and x else "")

            infer_df = infer_df.drop(columns=["fuzzy_inference"])

        else:
            infer_df[self.matched_company_name] = ""
            infer_df["Matched Company TFIDF"] = ""
            infer_df["Match Score"] = ""
            infer_df["Match Type"] = ""
            infer_df["Matched Company ID_CI"] = ""
            infer_df["Matched Company ISN ID"] = ""
            infer_df["Matched from Database Table"] = ""
            infer_df["Matched Demo Prospect Client"] = ""

        return infer_df

    def infer_fuzz(self, labels_dict2, refer_df):
        reference_df_copy = refer_df.copy()
        reference_df_copy[["fuzzy_score", "match_type"]] = reference_df_copy.apply(
            lambda x: self.calculate_fuzzy_score_new(x["labels_dict"], labels_dict2), axis=1, result_type="expand")

        sort_df = reference_df_copy[
            [self.refer_company_column, "labels_dict", "ID_CI", "CompanyID", "normalized_tfidf", "fuzzy_score",
             "match_type",
             "Database Table", "isDemo"]].sort_values(by=["fuzzy_score", "Database Table"],
                                                      ascending=[False, True]).head(1)
        # print(sort_df)
        return sort_df.to_dict(orient="records") if not sort_df.empty else {}

    def calculate_fuzzy_score_new(self, labels_dict1, labels_dict2):
        match_type = None
        subsidiary_score = 0
        generic_score = 0

        if type(labels_dict1) == str and type(labels_dict2) == str:
            labels_dict1 = ast.literal_eval(labels_dict1)
            labels_dict2 = ast.literal_eval(labels_dict2)

        # Getting the important keywords both the strings
        important_words1 = [key for key in labels_dict1.keys() if labels_dict1[key] == 0]
        important_words2 = [key for key in labels_dict2.keys() if labels_dict2[key] == 0]

        # An ordered match among the important keywords
        fuzzy_score = self.ordered_fuzzy_match(important_words1, important_words2)

        # If an important keywords match, then
        if fuzzy_score >= 85:
            subsidiary_words1 = [key for key in labels_dict1.keys() if labels_dict1[key] == 1]
            subsidiary_words2 = [key for key in labels_dict2.keys() if labels_dict2[key] == 1]

            generic_words1 = [key for key in labels_dict1.keys() if labels_dict1[key] == 2]
            generic_words2 = [key for key in labels_dict2.keys() if labels_dict2[key] == 2]

            # If no subsidiary keywords in both the strings, add the full subsidiary contribution to the score
            if not subsidiary_words1 and not subsidiary_words2:
                subsidiary_score += self.keyword_weights["1"] * 100

            # Same for the generic keywords
            if not generic_words1 and not generic_words2:
                generic_score += self.keyword_weights["2"] * 100

            # Unordered subsidiary match, if subsidiary keywords are present in either
            if subsidiary_words1 and subsidiary_words2:
                subsidiary_score += self.unordered_fuzzy_match(subsidiary_words1, subsidiary_words2,
                                                               factor=self.keyword_weights["1"] * 100)

            # Unordered generic match, if generic keywords are present in either
            if generic_words1 and generic_words2:
                generic_score += self.unordered_fuzzy_match(generic_words1, generic_words2,
                                                            factor=self.keyword_weights["2"] * 100)

            fuzzy_score += subsidiary_score + generic_score

            if fuzzy_score == 100:
                match_type = "Match"

            # If generic fuzzy score is less than 95, the two strings would be the aliases of the same company
            if generic_score < self.keyword_weights["2"] * 100 * 0.95:
                match_type = "Alias"

            # If subsidiary fuzzy score is less than 95, the two strings would be subsidiaries or sister companies
            if subsidiary_score < self.keyword_weights["1"] * 100 * 0.95:
                match_type = "Subsidiary"

        return fuzzy_score, match_type

    def ordered_fuzzy_match(self, l1, l2):
        max_len = max(len(l1), len(l2))
        total_score = 0
        total_weight = sum(1 / (i + 1) ** 2 for i in range(max_len))

        for i in range(max_len):
            if i < len(l1):
                elem1 = l1[i]
            else:
                elem1 = ""

            if i < len(l2):
                elem2 = l2[i]
            else:
                elem2 = ""

            # Calculate the fuzzy match score
            combined_score = self.fuzzy_phonetic_score(elem1, elem2)

            position_contribution = (85 / total_weight) * (1 / (i + 1) ** 2)
            weighted_score = combined_score * position_contribution / 100
            total_score += weighted_score

        return total_score

    def unordered_fuzzy_match(self, l1, l2, factor):
        # Ensure l1 is the smaller or equal list
        if len(l1) > len(l2):
            l1, l2 = l2, l1

        scores_matrix = self.calculate_match_scores(l1, l2)
        matches = self.find_best_matches(scores_matrix)
        total_score = self.calculate_total_score(matches, len(l2), factor=factor)

        return total_score

    @staticmethod
    def calculate_match_scores(l1, l2):
        l1_array = np.array(l1)
        l2_array = np.array(l2)

        # Create a DataFrame for all possible combinations
        df = pd.DataFrame([(a, b) for a in l1_array for b in l2_array], columns=['l1', 'l2'])

        # Calculate fuzzy scores using vectorized operations
        df['score'] = df.apply(
            lambda row: fuzz.ratio(row['l1'], row['l2']) if fuzz.ratio(row['l1'], row['l2']) >= 95 else 0, axis=1)

        # Pivot the DataFrame to create a matrix
        scores_matrix = df.pivot(index='l1', columns='l2', values='score').fillna(0).values

        return scores_matrix

    def fuzzy_phonetic_score(self, s1, s2):
        match_score = fuzz.ratio(s1, s2)
        phonetic_score = 0

        if s1 != "" and s2 != "" and 100 > match_score >= 90:
            # Calculate the phonetic similarity
            phonetic_score = fuzz.ratio(self.metaphone.phonetics(s1), self.metaphone.phonetics(s2))

        # Use the higher of the fuzzy match score and phonetic score
        combined_score = max(match_score, phonetic_score) if phonetic_score == 100 else match_score

        return combined_score

    @staticmethod
    def find_best_matches(scores_matrix):
        # Flatten the matrix and sort indices by score
        flat_indices = np.argsort(scores_matrix, axis=None)[::-1]

        matches = []
        used_rows, used_cols = set(), set()

        for flat_index in flat_indices:
            row, col = divmod(flat_index, scores_matrix.shape[1])

            if row not in used_rows and col not in used_cols:
                matches.append((row, col, scores_matrix[row, col]))
                used_rows.add(row)
                used_cols.add(col)

            if len(used_rows) == scores_matrix.shape[0] or len(used_cols) == scores_matrix.shape[1]:
                break

        return matches

    @staticmethod
    def calculate_total_score(matches, list_size, factor):
        total_score = sum(score for _, _, score in matches)

        # Normalize the score to be out of 10
        normalized_score = (total_score / (100 * list_size)) * factor

        return normalized_score

    def calculate_fuzzy_score(self, reference_list, test_list, labels_dict1, labels_dict2):
        total_fuzzy_score = 0
        # print(reference_list)
        # print(test_list)
        penalty = 0
        reference_list = ast.literal_eval(reference_list)
        test_list = ast.literal_eval(test_list)
        #     n = len(test)
        #     tfidf_matrix = np.tile(row, (n, 1))
        #     print(tfidf_matrix)

        list1 = np.array(reference_list)
        list2 = np.array(test_list)
        list1_expanded = np.repeat(list1[:, np.newaxis], list2.size, axis=1)
        list2_expanded = np.repeat(list2[np.newaxis, :], list1.size, axis=0)
        #     print(list1_expanded)
        #     print(list2_expanded)
        modified_fuzzy_score = partial(self.fuzzy_score, labels_dict1=labels_dict1, labels_dict2=labels_dict2)
        vectorized_modified_fuzzy_score = np.vectorize(modified_fuzzy_score, otypes=[float])

        fuzzy_matrix = np.transpose(vectorized_modified_fuzzy_score(list1_expanded, list2_expanded))
        # print("RAW FUZZY MATRIX")
        # print(fuzzy_matrix)

        argmax_idx = np.argmax(fuzzy_matrix, axis=0 if list1.size < list2.size else 1)
        argmax_idx2 = np.argmax(fuzzy_matrix, axis=0) if list1.size == list2.size else None

        if not np.all(np.diff(argmax_idx) > 0):
            # print("SASDf")
            #         print("WRONG!")
            #         print(argmax_idx)
            reference_sequence = np.arange(argmax_idx.size)
            deviations = np.sum(argmax_idx != reference_sequence)
            if list1.size == list2.size:
                argmax_idx = np.argmax(fuzzy_matrix, axis=1)

                # print(argmax_idx2)
                if not np.all(np.diff(argmax_idx) > 0):
                    #                 print("DISCARD")
                    deviations = np.sum(argmax_idx != reference_sequence)
            penalty -= deviations * 2.5

        mask = np.zeros_like(fuzzy_matrix)
        #     print("ARGMAX")
        #     print(argmax_idx)

        if list1.size < list2.size or (argmax_idx2 is not None):
            mask[argmax_idx, np.arange(fuzzy_matrix.shape[1])] = 1
            hot_fuzzy_matrix = fuzzy_matrix * mask
            argmax_idx_other = np.argmax(hot_fuzzy_matrix, axis=1)
            #         print(argmax_idx_other)
            mask = np.zeros_like(fuzzy_matrix)
            mask[np.arange(hot_fuzzy_matrix.shape[0]), argmax_idx_other] = 1
            hot_fuzzy_matrix = hot_fuzzy_matrix * mask
            #         print("HOT FUZZY MATRIX")
            #         print(hot_fuzzy_matrix)
            # tfidf2 = np.array(list(test_dict.values()))
            #         print("TFIDF2")
            #         print(tfidf2)
            #         print("MATMUL")
            #         print(np.matmul(tfidf2, hot_fuzzy_matrix))
            total_fuzzy_score = np.sum(hot_fuzzy_matrix) if argmax_idx2 is None else [
                np.sum(hot_fuzzy_matrix)]

        if list1.size >= list2.size or (argmax_idx2 is not None):
            # print("SDFS")
            mask[np.arange(fuzzy_matrix.shape[0]), argmax_idx] = 1
            hot_fuzzy_matrix = fuzzy_matrix * mask
            #         print(hot_fuzzy_matrix)
            argmax_idx_other = np.argmax(hot_fuzzy_matrix, axis=0)
            #         print(argmax_idx_other)
            mask = np.zeros_like(fuzzy_matrix)
            mask[argmax_idx_other, np.arange(hot_fuzzy_matrix.shape[1])] = 1
            hot_fuzzy_matrix = hot_fuzzy_matrix * mask
            #         print("HOT FUZZY MATRIX")
            #         print(hot_fuzzy_matrix)
            # tfidf1 = np.array(list(reference_dict.values()))
            #         print("TFIDF1")
            #         print(tfidf1)
            #         print("MATMUL")
            #         print(np.matmul(tfidf1, np.transpose(hot_fuzzy_matrix)))
            if argmax_idx2 is None:
                total_fuzzy_score = np.sum(np.transpose(hot_fuzzy_matrix))

            else:
                total_fuzzy_score.append(np.sum(np.transpose(hot_fuzzy_matrix)))

        #     score_matrix = np.multiply(tfidf_matrix, fuzzy_matrix)
        # #     print(score_matrix)
        #     total_fuzzy_score = np.sum(np.max(score_matrix, axis=0)) * 100

        # #     if list1.size < list2.size:
        # #         total_fuzzy_score -= 2 * (list2.size - list1.size)
        # print(total_fuzzy_score)
        # print(penalty)
        total_fuzzy_score = total_fuzzy_score if argmax_idx2 is None else sum(total_fuzzy_score) / 2

        # First keyword match logic
        # if total_fuzzy_score < 0.85 and list(reference_dict.keys())[0] == list(test_dict.keys())[0]:
        #     # total_fuzzy_score = 0.85
        #     if len(reference_dict) > len(test_dict):
        #         surplus = total_fuzzy_score - list(reference_dict.values())[0]
        #     elif len(reference_dict) < len(test_dict):
        #         surplus = total_fuzzy_score - list(test_dict.values())[0]
        #     else:
        #         surplus = total_fuzzy_score - (list(reference_dict.values())[0] + list(test_dict.values())[0]) / 2
        #
        #     total_fuzzy_score = 0.85 + surplus * 0.15
        # print(surplus)

        # print(penalty)
        # return (total_fuzzy_score * 100) + penalty if (total_fuzzy_score * 100) + penalty > 0 else 0
        return total_fuzzy_score * 100

    def fuzzy_score(self, s1, s2, labels_dict1={}, labels_dict2={}):
        # if (s1 == "S&C" or s1 == "ELECTRIC") and (s2 == "GENERAL" or s2 == "ELECTRIC"):
        # print("HEY!\nHEY\nHEY!" if s2 == "GENERAL" else "")
        # print("HEY!\nHEY\nHEY!" if s2 == "ELECTRIC" else "")
        # print(f"Reference word: {s1}")
        # print(f"Test word: {s2}")

        if type(labels_dict1) == str and type(labels_dict2) == str:
            labels_dict1 = ast.literal_eval(labels_dict1)
            labels_dict2 = ast.literal_eval(labels_dict2)

        # print("Reference Dict")
        # print(labels_dict1)
        # print("Test Dict")
        # print(labels_dict2)

        label1 = labels_dict1[s1]  # reference, more accurate predictions
        label2 = labels_dict2[s2]  # test

        # print(f"Label 1: {label1}")
        # print(f"Label 2: {label2}")

        label_counts1, label_counts2 = {}, {}

        for i in range(0, 3):
            label_counts1.update({str(i): self.count_value_in_dict(labels_dict1, i)})
            label_counts2.update({str(i): self.count_value_in_dict(labels_dict2, i)})

        # print(f"Label counts 1: {label_counts1}")
        # print(f"Label counts 2: {label_counts2}")

        fuzz_ratio = fuzz.ratio(s1, s2) / 100 if fuzz.ratio(s1, s2) >= 95 else 0

        # print(f"fuzz ratio: {fuzz_ratio}")

        if fuzz_ratio >= 0.95:
            if len(labels_dict2) == 1:
                if label_counts1["0"] == 1 and label_counts1["1"] == 0 and label_counts1["2"] == 0:
                    fuzz_ratio = fuzz_ratio
                elif label_counts1["0"] == 1 and label_counts1["1"] == 0 and label_counts1["2"] != 0:
                    fuzz_ratio = fuzz_ratio * (self.keyword_weights["0"] + self.keyword_weights["1"])
                elif label_counts1["0"] == 1 and label_counts1["1"] != 0 and label_counts1["2"] == 0:
                    fuzz_ratio = fuzz_ratio * (self.keyword_weights["0"] + self.keyword_weights["2"])

            elif label1 == label2:
                # print(f"Equal label: {label1}")
                if label_counts1[str(label1)] >= label_counts2[str(label1)]:
                    fuzz_ratio = fuzz_ratio * (self.keyword_weights[str(label1)] / label_counts1[str(label1)])
                    # print(f"fuzz ratio mod: {fuzz_ratio}")
                else:
                    fuzz_ratio = fuzz_ratio * (self.keyword_weights[str(label1)] / label_counts2[str(label1)])
                    # print(f"fuzz ratio mod: {fuzz_ratio}")

            elif (label1 == 0 and label2 == 1) or (label1 == 1 and label2 == 0):
                if label1 == 0:
                    fuzz_ratio = fuzz_ratio * (self.keyword_weights[str(label1)] / label_counts1[str(label1)])
                    # print(f"fuzz ratio mod: {fuzz_ratio}")
                elif label2 == 0:
                    fuzz_ratio = fuzz_ratio * (self.keyword_weights[str(label2)] / label_counts2[str(label2)])
                    # print(f"fuzz ratio mod: {fuzz_ratio}")

        return fuzz_ratio

        # else:
        #     return 0

    @staticmethod
    def count_value_in_dict(d, value):
        return sum(1 for v in d.values() if v == value)


class ArgParser(KFoldTFIDFArgParser):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Argument Parser for TFIDF Fuzzy Matching")
        self.add_arguments()

        self.parser.add_argument("--np2", "--n_process2", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8], default=4,
                                 help="Number of processes while multiprocessing 2")

        self.args = None
        self.parse()


if __name__ == "__main__":
    start = time.time()
    mp.freeze_support()

    args = ArgParser().args
    df = pd.read_json(args["i"], orient="records")
    tfidf_fuzzy_matching = TFIDFFuzzyMatching(df)
    print(tfidf_fuzzy_matching.reference_df)

    # Reference data is first only subscribed HCs
    # k + l (all infer df) -> m (only subscribed HCs)
    tfidf_fuzzy_matching.reference_df = df[(df["Source"] == "ISN Database") & (df["HasOwnerRole"] == 1) &
                                           (df["Database Table"] != "DMSalesforce.dim.SalesforceAccount")]

    print(tfidf_fuzzy_matching.infer_df)
    print(tfidf_fuzzy_matching.reference_df)

    result_df = tfidf_fuzzy_matching.match_multiprocess(args["np"])
    print(result_df)

    # Those input records that did not match with subscribed are retained and considered for ISNetworld Prospect matches
    # l (low confidence HC matches) -> n (only HC prospects)
    prospect_infer_df = result_df[(result_df["Source"] == "Salesforce") & (result_df["Match Score"] < 85)][
        ["RecordID_CI", "AHC Company Name", "Source", "HasOwnerRole", "ID_CI", "CompanyID", "HasContractorRole", "Role",
         "Database Table", "isDemo", "CompanyNameCleaned", "words", "cleaned_words", "labels_dict", "term_frequency",
         "tfidf", "normalized_tfidf", "Count", "Report Column"]]
    tfidf_fuzzy_matching.infer_df = prospect_infer_df
    tfidf_fuzzy_matching.reference_df = df[(df["Source"] == "ISN Database") & (df["HasOwnerRole"] == 1) &
                                           (df["Database Table"] == "DMSalesforce.dim.SalesforceAccount")]

    print(tfidf_fuzzy_matching.infer_df)
    print(tfidf_fuzzy_matching.reference_df)

    result_df2 = tfidf_fuzzy_matching.match_multiprocess(args["np2"])

    result_df2[result_df2["Match Score"] < 85]["Matched Demo Prospect Client"] = 0 # double check

    final_result_df = pd.concat([result_df[result_df["Match Score"] >= 85], result_df2])
    # final_result_df = result_df

    # subscriber_prospect_hc_matches = final_result_df[final_result_df["Match Score"] >= 85]
    # non_subscriber_hc_matches = final_result_df[final_result_df["Match Score"] < 85]

    # Input data being matched with itself for Company grouping
    tfidf_fuzzy_matching.refer_company_column = "AHC Company Name"
    tfidf_fuzzy_matching.matched_company_name = "Standardized Alias"
    tfidf_fuzzy_matching.infer_df = final_result_df
    tfidf_fuzzy_matching.reference_df = final_result_df
    tfidf_fuzzy_matching.self_match = True

    final_result_df = tfidf_fuzzy_matching.match_multiprocess(args["np"])

    final_result_df.to_json(args["o"], orient="records")

    print(final_result_df)

    print(f"Time taken: {time.time() - start} seconds")
