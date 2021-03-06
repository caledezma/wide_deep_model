"""
Module containing the functions required to train a wide and deep model.
"""
import argparse
import pickle
import tensorflow as tf
import tqdm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

DESCRIPTION = "description"

def load_data(dataset_path, feature_field, target_field):
    """
    Loads the kaggle wine dataset with the desired target feature.
    """
    dataset = pd.read_csv(
        dataset_path
    )[[feature_field, target_field]].dropna(
        axis=0,
        how="any"
    ).drop_duplicates(subset=feature_field, keep="first")
    return dataset[feature_field].tolist(), dataset[target_field].tolist()

def process_data(text_feature, count_vec=None, vec_path=None, vocab_size=2000, doc_length_est=500):
    """
    Processes the text feature using the count vectoriser and returns it in a format that is
    suitable for training a wide and deep model
    """
    print("Calculating inputs to wide model")
    if not count_vec:
        count_vec = CountVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            max_features=vocab_size,
        )
        wide_inputs = (count_vec.fit_transform(raw_documents=text_feature) > 0).astype(int)
    else:
        wide_inputs = (count_vec.transform(raw_documents=text_feature) > 0).astype(int)
    vocab = count_vec.get_feature_names()
    unk = len(vocab) + 1
    analyse = count_vec.build_analyzer()
    deep_inputs = np.zeros((len(text_feature), doc_length_est), dtype=int)
    max_doc_length = 0
    for idx, document in tqdm.tqdm(
            enumerate(text_feature),
            total=len(text_feature),
            desc="Transforming text features for deep model",
        ):
        tokenised_document = analyse(document)
        num_words = len(tokenised_document)
        max_doc_length = num_words if num_words > max_doc_length else max_doc_length
        deep_input = np.array(
            [vocab.index(word)+1 if word in vocab else unk for word in tokenised_document]
        )
        if deep_input.size > deep_inputs.shape[1]:
            deep_inputs = np.hstack(
                (
                    deep_inputs,
                    np.zeros((len(text_feature), deep_input.size - deep_inputs.shape[1]))
                )
            )
        deep_inputs[idx, :deep_input.size] = deep_input
    if max_doc_length < doc_length_est:
        deep_inputs = deep_inputs[:, :max_doc_length]

    if vec_path:
        print("Saving vectoriser")
        pickle.dump(count_vec, open(vec_path, "wb"))
    return wide_inputs, deep_inputs

def get_wide_deep_model(
    num_wide_features,
    num_deep_features,
    embedding_size=64,
    num_layers=2,
    n_units=512,
    dropout_prob=0.2,
    learning_rate=1e-3,
    lr_decay=1e-6,
    reg_coeff=1e-5,
    **kwargs
):
    """
    Returns a compiled wide and deep model with the desired target type.
    """
    wide_input = tf.keras.layers.Input(shape=(num_wide_features,))
    deep_input = tf.keras.layers.Input(shape=(num_deep_features,))
    model_output = tf.keras.layers.Embedding(
        num_wide_features+2,
        embedding_size,
    )(deep_input)
    model_output = tf.keras.backend.sum(model_output, axis=1)
    for _ in range(num_layers):
        model_output = tf.keras.layers.Dense(
            units=n_units,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(float(reg_coeff)),
        )(model_output)
        model_output = tf.keras.layers.Dropout(rate=dropout_prob)(model_output)
        model_output = tf.keras.layers.BatchNormalization()(model_output)

    model_output = tf.keras.layers.concatenate([wide_input, model_output])
    model_output = tf.keras.layers.Dense(
        units=1,
        kernel_regularizer=tf.keras.regularizers.l2(float(reg_coeff))
    )(model_output)
    model = tf.keras.Model([wide_input, deep_input], model_output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(learning_rate),
            decay=float(lr_decay),
        ),
        loss="mse",
    )
    return model

def parse_cli_args():
    """
    Parses the CLI arguments required to train and demo the ML models
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--data-path",
        help="Full path to where the dataset is located.",
        type=str,
        required=True,
    )
    arg_parser.add_argument(
        "--features-field",
        help="Name of the field (in the dataset) that contains the training features",
        type=str,
        default="description"
    )
    arg_parser.add_argument(
        "--target-field",
        help="Name of the field (in the dataset) that contains the targets to be used for training",
        type=str,
        default="points",
    )
    arg_parser.add_argument(
        "--model-config",
        help="Full path to the yaml file containing the model details.",
        type=str,
        required=True,
    )
    arg_parser.add_argument(
        "--model-path",
        help="Full path to where the ML model should be saved. The mdoel is saved as tf.keras H5",
        type=str,
        required=True,
    )
    arg_parser.add_argument(
        "--vectoriser-path",
        help="Full path to where the text vectoriser should be saved. The vectoriser is saved as a"
            " pickled sklearn CountVectorizer",
        type=str,
        required=True,
    )

    return arg_parser.parse_args()