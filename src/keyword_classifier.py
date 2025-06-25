import argparse
import copy
import ast
import pandas as pd
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Bidirectional, Attention, Dropout, \
    BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow import math
import numpy as np
import time
import re
import os
from src.kfold_tfidf_generator import ArgParser as KFoldTFIDFArgParser


class KeywordClassifier:
    def __init__(self, data: pd.DataFrame, mode="test",
                 model="keyword_classifier.weights.h5"):
        self.data = data
        self.mode = mode
        # Neural Network specific parameters
        self.embedding_dim = 300
        self.sequence_length = 10
        self.num_pos_tags = 19

        # Important, Subsidiary, Generic, None
        self.num_classes = 4

        self.model = None
        self.compile_model()
        self.X_embeddings, self.X_tfidf, self.X_pos, self.X_position, self.y = None, None, None, None, None
        self.early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        self.lr_scheduler = LearningRateScheduler(
            lambda epoch, lr: float(lr * math.exp(-0.01)) if epoch > 10 else float(lr))
        self.models_directory = "models/"

        self.data_pred = pd.DataFrame()

        self.model_save_path = self.models_directory + f"ml_fuzzy_matching_{int(time.time())}.weights.h5"
        self.model_load_path = self.models_directory + model

    def run(self):
        self.preprocess_data()

        if self.mode == "train":
            self.train()

        self.predict()

    # @staticmethod
    # def safe_lit(x):
    #     return ast.literal_eval(x) if isinstance(x, str) else x

    @staticmethod
    def safe_lit(x):
        # If it’s already parsed (not a str), just return it
        if not isinstance(x, str):
            return x

        # 1) Try pure literal_eval first (for plain lists, dicts, numbers…)
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            pass

        # 2) Fallback: use a very restricted eval that only knows about numpy.array
        #    and the dtype name float32
        #    This will safely parse array([...], dtype=float32) syntax.
        return eval(
            x,
            {"__builtins__": None},  # no built‑ins
            {"array": np.array, "float32": np.float32}
        )


    def preprocess_data(self):
        # Feature encoding
        for col in ["tfidf", "pos_num", "word_vecs", "words"]:
            def try_lit(x):
                try:
                    return self.safe_lit(x)
                except Exception as e:
                    # Log column name, row index, and offending value
                    # (you could use logging.debug/info here instead of print)
                    print(
                        f"Error parsing column={col!r}, index={x.name if hasattr(x, 'name') else 'unknown'}, value={x!r}")
                    # re-raise so you see the full traceback
                    raise

            # apply with our wrapped version
            # note: when you use Series.apply, the function sees just the value, so to capture the index:
            # you can do `for idx, val in self.data[col].items(): ...` instead—see alternate below
            self.data[col] = self.data[col].apply(try_lit)
        #
        # self.data["tfidf"] = self.data["tfidf"].apply(lit)
        # self.data["pos_num"] = self.data["pos_num"].apply(lit)
        # self.data["word_vecs"] = self.data["word_vecs"].apply(lit)
        # self.data["words"] = self.data["words"].apply(lit)
        print(len(self.data))

        # Offsetting to account for one-hot encoding
        self.data["pos_num_modified"] = self.data["pos_num"].apply(lambda x: [int(i) + 1 for i in x])
        self.data["tfidf_values"] = self.data["tfidf"].apply(lambda x: [float(tfidf) for tfidf in x.values()])

        # Restricting the features to max sequence length
        self.data["pos_num_modified"] = self.data["pos_num_modified"].apply(lambda x: x[:self.sequence_length])
        self.data["tfidf_values"] = self.data["tfidf_values"].apply(lambda x: x[:self.sequence_length])
        self.data["word_vecs"] = self.data["word_vecs"].apply(lambda x: [np.array(i) for i in x][:self.sequence_length])

        tfidf_values = []

        self.data["tfidf_values"].apply(lambda x: [tfidf_values.append(i) for i in x])

        # TFIDF batch normalization
        max_tfidf = max(tfidf_values)
        min_tfidf = min(tfidf_values)

        self.data["tfidf_minmax_normalized"] = self.data["tfidf_values"].apply(
            lambda x: [(tfidf - min_tfidf) / (max_tfidf - min_tfidf) for tfidf in x])

        embeddings_list = self.data["word_vecs"].tolist()
        tfidf_list = self.data["tfidf_minmax_normalized"].apply(lambda x: [[v] for v in x]).tolist()
        pos_list = self.data["pos_num_modified"].apply(lambda x: [[v] for v in x]).tolist()
        position_list = self.data["words"].apply(
            lambda x: [[i + 1] for i in range(len(x[:self.sequence_length]))]).tolist()

        # Padding upto max sequence length
        padding_embedding = [0.0] * self.embedding_dim
        padding_tfidf = [0.0]
        padding_pos = [0.0]
        padding_position = [0.0]
        padding_label = 3

        self.X_embeddings = self.pad_sequences(embeddings_list, self.sequence_length, padding_embedding)
        self.X_tfidf = self.pad_sequences(tfidf_list, self.sequence_length, padding_tfidf)
        self.X_pos = self.pad_sequences(pos_list, self.sequence_length, padding_pos)
        self.X_position = self.pad_sequences(position_list, self.sequence_length, padding_position)

        print(self.X_embeddings.shape)
        print(self.X_tfidf.shape)
        print(self.X_pos.shape)
        print(self.X_position.shape)

        if self.mode == "train":
            labels_list = self.data["label_list"].apply(lambda x: [v for v in x]).tolist()
            y_padded = self.pad_sequences(labels_list, self.sequence_length, padding_label)
            self.y = self.one_hot_encode_labels(y_padded, self.num_classes)
            print(self.y.shape)

    def predict(self):
        if self.mode == "test":
            print(f"⏱ loading weights from -> {os.path.abspath(self.model_load_path)}")
            print(f"⏱ directory contents: {os.listdir(self.models_directory)}")
            self.model.load_weights(self.model_load_path)

        predictions = self.model.predict([self.X_embeddings, self.X_tfidf,
                                          self.X_pos,
                                          self.X_position], verbose=0)

        predicted_labels = np.argmax(predictions, axis=-1)

        if self.mode == "train":
            self.test(predicted_labels=predicted_labels)

        else:
            # Label decoding
            data_pred = []
            label_map = {0: "Important", 1: "Subsidiary", 2: "Generic", 3: "None"}
            predicted_labels_mapped = [[label_map[label] for label in sequence] for sequence in predicted_labels]

            for i, record in enumerate(self.data.to_dict(orient="records")):
                new_record = record
                filtered_labels = [s for s in predicted_labels_mapped[i] if s != "None"]
                predicted_labels_list = [str(list(label_map.keys())[list(label_map.values()).index(label)]) for label in
                                         filtered_labels]
                if predicted_labels_list:
                    new_record.update(
                        {"Predicted Labels": ", ".join(predicted_labels_list) if len(predicted_labels_list) > 1 else
                        predicted_labels_list[0] + ", "})
                else:
                    new_record.update({"Predicted Labels": ""})
                new_record.update({"Predicted Label Names": filtered_labels})
                data_pred.append(new_record)

            self.data_pred = pd.DataFrame(data_pred)
            self.data_pred["label_list"] = self.data_pred["Predicted Labels"].apply(
                lambda x: [int(item.strip().replace(",", "")) for item in x.split(", ") if item])

            self.data_pred["cleaned_words"] = self.data_pred["words"].apply(lambda x: [word.upper() for word in x])
            self.data_pred["cleaned_words"] = self.data_pred["cleaned_words"].apply(
                lambda x: [re.sub(r"[^A-Za-z0-9\&]", "", i) for i in x])
            self.data_pred["labels_dict"] = self.data_pred.apply(
                lambda x: self.get_labels_dict(x), axis=1)

    @staticmethod
    def get_labels_dict(x):
        # If lesser labels and more input words, mark the unlabeled words as Important
        result = dict(zip(x["cleaned_words"], x["label_list"]))
        cleaned_words = copy.deepcopy(x["cleaned_words"])

        if len(x["cleaned_words"]) > len(x["label_list"]):
            for i in result.keys():
                cleaned_words.remove(i)

            for key in cleaned_words:
                result.update({key: 0})

        return result

    def test(self, predicted_labels):
        true_labels = np.argmax(self.y, axis=-1)
        print(f"Non-padding accuracy: {self.calculate_non_padding_accuracy(true_labels, predicted_labels)}")

    def train(self):
        # Perform batch (instead of stochastic) training for better results
        history = self.model.fit([self.X_embeddings, self.X_tfidf,
                                  self.X_pos,
                                  self.X_position], self.y, batch_size=1024, epochs=500, validation_split=0.15,
                                 callbacks=[self.early_stopping, self.lr_scheduler])
        self.model.save_weights(self.model_save_path)

    @staticmethod
    def calculate_non_padding_accuracy(y_true, y_pred, padding_label=3):
        # Calculate accuracy for Important, Subsidiary and Generic classes
        correct = 0
        total = 0

        for true_seq, pred_seq in zip(y_true, y_pred):
            for true_label, pred_label in zip(true_seq, pred_seq):
                if true_label != padding_label:
                    if true_label == pred_label:
                        correct += 1
                    total += 1

        accuracy = correct / total if total > 0 else 0
        return accuracy

    @staticmethod
    def one_hot_encode_labels(padded_labels, num_classes):
        one_hot_labels = []
        for sequence in padded_labels:
            one_hot_sequence = []
            for label in sequence:
                one_hot_sequence.append(to_categorical(label, num_classes=num_classes))
            one_hot_labels.append(one_hot_sequence)

        return np.array(one_hot_labels)

    @staticmethod
    def min_max_normalize(tensor):
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        return (tensor - min_val) / (max_val - min_val)

    @staticmethod
    def pad_sequences(input_data, max_length, padding_value):
        padded_data = []
        for sequence in input_data:
            if len(sequence) < max_length:
                sequence = sequence + [padding_value] * (max_length - len(sequence))
            padded_data.append(sequence)
        return np.array(padded_data)

    def compile_model(self):
        embedding_input = Input(shape=(self.sequence_length, self.embedding_dim), name="embedding_input")
        tfidf_input = Input(shape=(self.sequence_length, 1), name="tfidf_input")
        pos_input = Input(shape=(self.sequence_length, 1), name="pos_input")
        position_input = Input(shape=(self.sequence_length, 1), name="position_input")

        lstm_out = Bidirectional(LSTM(64, return_sequences=True, name="lstm"))(embedding_input)
        lstm_out = BatchNormalization()(lstm_out)

        attention = Attention()([lstm_out, lstm_out])
        concat = Concatenate()([lstm_out, attention])
        lstm_out = LSTM(64, return_sequences=True)(concat)

        combined = Concatenate()([lstm_out, tfidf_input, pos_input, position_input])
        combined = Dropout(0.1)(combined)

        dense = Dense(64, activation="relu")(combined)
        output = Dense(self.num_classes, activation="softmax")(dense)

        self.model = Model(inputs=[embedding_input, tfidf_input,
                                   pos_input,
                                   position_input], outputs=output)

        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        self.model.summary()


class ArgParser(KFoldTFIDFArgParser):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Argument Parser for TFIDF Fuzzy Matching")
        self.add_arguments()

        self.parser.add_argument("--m", "--mode", type=str, choices=["test", "train"], default="test",
                                 help="Keyword Classifier mode: Test or Train")
        self.parser.add_argument("--mn", "--model_name", type=str, default="ml_matching9_model.weights.h5",
                                 help="Keyword Classifier test model name")

        self.args = None
        self.parse()


if __name__ == "__main__":
    start = time.time()

    args = ArgParser().args
    df = pd.read_json(args["i"], orient="records")
    keyword_classifier = KeywordClassifier(df, mode=args["m"], model=args["mn"])
    print(keyword_classifier.data)

    keyword_classifier.run()
    result_df = keyword_classifier.data_pred
    result_df.to_json(args["o"], orient="records")

    print(result_df)

    print(f"Time taken: {time.time() - start} seconds")
