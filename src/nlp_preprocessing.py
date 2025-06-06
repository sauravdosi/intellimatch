import argparse
import time
import multiprocessing as mp
import spacy
import pandas as pd
from configparser import ConfigParser
from src.tfidf_preprocessing import TFIDFPreprocessing
from src.kfold_tfidf_generator import ArgParser as KFoldTFIDFArgParser
spacy.require_gpu()

class NLPPreprocessing(TFIDFPreprocessing):
    # Feature Engineering except for TFIDF

    def __init__(self, df, n_process=3, config_path="config/config.ini"):
        super().__init__(df, n_process=n_process)
        self.config = ConfigParser()
        self.config.read([config_path])
        self.word_vectors_spacy_model = self.config.get("NLP_PREPROCESSING", "word_vectors_spacy_model")
        self.pos_spacy_model = self.config.get("NLP_PREPROCESSING", "pos_spacy_model")
        self.input_column = self.config.get("NLP_PREPROCESSING", "input_column")
        self.output_column_vecs = self.config.get("NLP_PREPROCESSING", "output_column_vecs")
        self.output_column_pos = self.config.get("NLP_PREPROCESSING", "output_column_pos")

        self.nlp_lg = spacy.load(self.word_vectors_spacy_model)
        self.nlp_trf = spacy.load(self.pos_spacy_model)

        self.nlp_lg.disable_pipe("parser")
        self.nlp_lg.disable_pipe("lemmatizer")
        self.nlp_lg.disable_pipe("ner")
        self.nlp_lg.disable_pipe("tagger")
        self.nlp_lg.disable_pipe("attribute_ruler")

        self.nlp_trf.disable_pipe("parser")
        self.nlp_trf.disable_pipe("lemmatizer")
        self.nlp_trf.disable_pipe("ner")

        # Encoders for POS output
        self.pos_mapping = {
            "": 0,
            "ADJ": 1,
            "ADP": 2,
            "ADV": 3,
            "AUX": 4,
            "CCONJ": 5,
            "DET": 6,
            "INTJ": 7,
            "NOUN": 8,
            "NUM": 9,
            "PART": 10,
            "PRON": 11,
            "PROPN": 12,
            "PUNCT": 13,
            "SCONJ": 14,
            "SYM": 15,
            "VERB": 16,
            "X": 17,
            "SPACE": 18
        }

    def preprocess(self, data):
        # 1) Clean names and split into words
        data["CompanyNameCleaned"] = (
            data[self.input_column]
            .str.replace(" - ", " ", regex=False)
        )
        data["words"] = data["CompanyNameCleaned"].str.split(" ")

        # 2) Prepare list of word-lists for batching
        word_lists = data["words"].tolist()

        # 3) Batch-process through GPU-enabled pipelines
        #    (make sure you called spacy.require_gpu() in __init__ or before)
        #    We use n_process=1 because GPU handles parallelism internally.
        #    Adjust batch_size up or down until you hit your GPU’s sweet spot.
        docs_lg = list(self.nlp_lg.pipe(
            (w for wl in word_lists for w in wl),
            batch_size=1024,  # words per batch
            n_process=1))
        docs_trf = list(self.nlp_trf.pipe(
            (w for wl in word_lists for w in wl),
            batch_size=512,
            n_process=1))

        # 4) Re-chunk flat docs back into per-row lists
        vecs, pos_dicts = [], []
        it_lg = iter(docs_lg)
        it_trf = iter(docs_trf)
        for wl in word_lists:
            # for each word-list, pull that many docs out
            docs_chunk_lg = [next(it_lg) for _ in wl]
            docs_chunk_trf = [next(it_trf) for _ in wl]

            # vectors: one array per word
            vecs.append([doc.vector for doc in docs_chunk_lg])

            # ner+pos: run your existing logic but on docs_chunk_trf
            ner_list, pos_list = [], []
            for doc in docs_chunk_trf:
                # replicate extract_ner_pos behavior exactly:
                done = False
                for token in doc:
                    if not done:
                        if token.text == "Corp." or token.pos_ != "PUNCT":
                            if token.text == "Corp.":
                                pos_list.append("NOUN")
                            else:
                                if "-" in token.text:
                                    if token.pos_ == "PROPN":
                                        pos_list.append(token.pos_)
                                        done = True
                                else:
                                    pos_list.append(token.pos_)
                if not doc:
                    pos_list.append("")
                ner_list.append([])  # original code didn’t populate ner

            pos_dicts.append({"ner": ner_list, "pos": pos_list})

        # 5) Assign back into DataFrame columns
        data[self.output_column_vecs] = vecs
        data["ner_pos_dict"] = pos_dicts
        data["ner"] = [d["ner"] for d in pos_dicts]
        data["pos"] = [d["pos"] for d in pos_dicts]
        data[self.output_column_pos] = [
            [self.pos_mapping[p] for p in row] for row in data["pos"]
        ]

        return data

    # def preprocess(self, data):
    #     data["CompanyNameCleaned"] = data[self.input_column].apply(lambda x: x.replace(" - ", " "))
    #     data["words"] = data["CompanyNameCleaned"].apply(lambda x: x.split(" "))
    #     # Word Embeddings from the Large model
    #     data[self.output_column_vecs] = data["words"].apply(lambda x: [self.nlp_lg(word).vector for word in x])
    #     # NER and POS from the Transformer model
    #     data["ner_pos_dict"] = data["words"].apply(lambda x: self.extract_ner_pos(x))
    #     data["ner"] = data["ner_pos_dict"].apply(lambda x: x["ner"])
    #     data["pos"] = data["ner_pos_dict"].apply(lambda x: x["pos"])
    #     data[self.output_column_pos] = data["pos"].apply(lambda x: [self.pos_mapping[pos] for pos in x])
    #
    #     return data

    def extract_ner_pos(self, word_list):
        ner_pos_dict = {"ner": [], "pos": []}

        for word in word_list:
            doc = self.nlp_trf(word)
            done = False

            # if doc.ents:
            #     for ent in doc.ents:
            #         ner_pos_dict["ner"].append(ent.label_)
            # else:
            #     ner_pos_dict["ner"].append("")

            if doc:
                for token in doc:
                    if not done:
                        if token.text == "Corp." or token.pos_ != "PUNCT":
                            if token.text == "Corp.":
                                ner_pos_dict["pos"].append("NOUN")
                            else:
                                if "-" in word:
                                    if token.pos_ == "PROPN":
                                        ner_pos_dict["pos"].append(token.pos_)
                                        done = True
                                else:
                                    ner_pos_dict["pos"].append(token.pos_)

            else:
                ner_pos_dict["pos"].append("")

        return ner_pos_dict


class ArgParser(KFoldTFIDFArgParser):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Argument Parser for TFIDF Fuzzy Matching")
        self.add_arguments()
        self.args = None
        self.parse()


if __name__ == "__main__":
    start = time.time()
    mp.freeze_support()

    args = ArgParser().args
    df = pd.read_json(args["i"], orient="records")
    nlp_preprocessor = NLPPreprocessing(df, n_process=args["np"])
    print(nlp_preprocessor.df)

    nlp_preprocessor.multiprocess_preprocess()
    result_df = nlp_preprocessor.result_df

    result_df.to_json(args["o"], orient="records")

    print(result_df)

    print(f"Time taken: {time.time() - start} seconds")