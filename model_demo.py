"""
Demo the model at test time
"""
import os
import pickle
import yaml
import numpy as np

from utils import load_wine_data, get_wide_deep_model, process_data

MODEL_CONFIG = ["model_config/model_config_1.yaml", "model_config/model_config_2.yaml"]
DATA_PATH = "wine_data/wine_dataset.csv"
MODEL_PATH = ["saved_models/model_1.h5", "saved_models/model_2.h5"]
VEC_PATH = ["saved_models/count_vec_2.pkl", "saved_models/count_vec_2.pkl"]

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supress TF warnings
    print("Loading data...")
    X, y = load_wine_data(DATA_PATH, "points")
    random_idx = (np.random.rand(5)*len(X)).astype(int)
    X = [X[idx] for idx in random_idx]
    y = [y[idx] for idx in random_idx]

    X_wide_deep = [
        process_data(X, count_vec=pickle.load(open(vectoriser, "rb"))) for vectoriser in VEC_PATH
    ]

    print("Constructing Keras models...")
    prediction_models = [
        get_wide_deep_model(
            num_wide_features=X[0].shape[1],
            num_deep_features=X[1].shape[1],
            **yaml.safe_load(open(model_conf, "r")),
        ) for X, model_conf in zip(X_wide_deep, MODEL_CONFIG)]

    for weights, model in zip(MODEL_PATH, prediction_models):
        model.load_weights(weights)

    print("Predicting...")
    predictions = [
            model.predict([X[0], X[1]], verbose=0)\
                for X, model in zip(X_wide_deep, prediction_models)
    ]

    for pred_idx, description, target in zip(range(len(X)), X, y):
        print("="*100)
        print("Wine review:\n", description)
        print("Reviewer score:", target)
        for model_idx in range(len(prediction_models)):
            print("Model", model_idx, "prediction:", predictions[model_idx][pred_idx])

if __name__ == "__main__":
    main()