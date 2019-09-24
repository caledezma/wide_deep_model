"""
Demo the model at test time
"""
import yaml
import numpy as np

from utils import load_wine_data, process_data, get_wide_deep_model

MODEL_CONFIG = "model_config/model_config_1.yaml"
DATA_PATH = "wine_data/wine_dataset.csv"
MODEL_PATH = "saved_models/model_1.h5"

def main():
    print("Loading data...")
    X, y = load_wine_data(DATA_PATH, "points")
    random_idx = (np.random.rand(5)*len(X)).astype(int)
    X, y = X[random_idx], y[random_idx]
    X_wide, X_deep = process_data(X)


    print("Constructing Keras model...")
    model_config = yaml.safe_load(open(MODEL_CONFIG, "r"))
    wide_deep_model = get_wide_deep_model(
        num_wide_features=X_wide.shape[1],
        num_deep_features=X_deep.shape[1],
        **model_config,
    )
    wide_deep_model.load_weights(MODEL_PATH)

    print("Predicting...")
    model_predictions = wide_deep_model.predict([X_wide, X_deep], verbose=0)

    for description, prediction, target in zip(X, model_predictions, y):
        print("="*100)
        print("Wine review:\n", description)
        print("Reviewer score:", target)
        print("Model prediction:", prediction)

if __name__ == "__main__":
    main()