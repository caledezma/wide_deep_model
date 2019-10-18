"""
Demo the model at test time
"""
import os
import pickle
import yaml
import numpy as np

from utils import load_data, get_wide_deep_model, process_data, parse_cli_args

MODEL_CONFIG = [
    "model_config/model_config_1.yaml",
    "model_config/model_config_2.yaml",
    "model_config/model_config_3.yaml",
    "model_config/model_config_4.yaml",
]
DATA_PATH = "wine_data/wine_dataset.csv"
MODEL_PATH = [
    "saved_models/model_1.h5",
    "saved_models/model_2.h5",
    "saved_models/model_3.h5",
    "saved_models/model_4.h5",
]
VEC_PATH = [
    "saved_models/count_vec_1.pkl",
    "saved_models/count_vec_2.pkl",
    "saved_models/count_vec_3.pkl",
    "saved_models/count_vec_4.pkl",
]

def main():
    """Main block of code. Loads the data, model and vectoriser and shows a demo"""
    args = parse_cli_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supress TF warnings

    print("Loading data...")
    X, y = load_data(
        dataset_path=args.data_path,
        feature_field=args.features_field,
        target_field=args.target_field,
    )
    # Choose five random examples to show
    random_idx = (np.random.rand(5)*len(X)).astype(int)
    X = [X[idx] for idx in random_idx]
    y = [y[idx] for idx in random_idx]

    X_wide_deep = process_data(
        text_feature=X,
        count_vec=pickle.load(open(args.vectoriser_path, "rb")),
    )

    print("Constructing Keras model")
    prediction_model = get_wide_deep_model(
        num_wide_features=X_wide_deep[0].shape[1],
        num_deep_features=X_wide_deep[1].shape[1],
        **yaml.safe_load(open(args.model_config, "r")),
    )
    prediction_model.load_weights(args.model_path)
    print("Predicting...")
    predictions = prediction_model.predict([X_wide_deep[0], X_wide_deep[1]], verbose=0)

    for prediction, text, target in zip(predictions, X, y):
        print("="*100)
        print("Text:\n", text)
        print("Target:", target)
        print("Model's prediction:", prediction)

if __name__ == "__main__":
    main()