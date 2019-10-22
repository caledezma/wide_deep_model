"""
Model training module
"""
import os
import pickle
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from utils import load_data, process_data, get_wide_deep_model, parse_cli_args

def main():
    """Main block of code. Reads the data, constructs the tokeniser and trains the model"""
    args = parse_cli_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Supress TF warnings
    X, y = load_data(
        dataset_path=args.data_path,
        feature_field=args.features_field,
        target_field=args.target_field,
    )

    y=np.array(y)

    print("Dataset loaded {0} examples".format(len(X)))
    model_config = yaml.safe_load(open(args.model_config, "r"))
    X_wide, X_deep = process_data(
        text_feature=X,
        vec_path=args.vectoriser_path,
        vocab_size=model_config["vocab_size"]
    )
    X_wide_train, X_wide_test, X_deep_train, X_deep_test, y_train, y_test =\
        train_test_split(X_wide, X_deep, y, test_size=0.2)

    print(
        "Train data contains",
        y_train.shape[0],
        "examples and test data contains",
        y_test.shape[0],
        "examples"
    )

    print("Constructing Keras model")
    model = get_wide_deep_model(
        num_wide_features=X_wide.shape[1],
        num_deep_features=X_deep_train.shape[1],
        **model_config,
    )

    print("Training...")
    model.fit(
        x=[X_wide_train, X_deep_train],
        y=y_train,
        epochs=model_config["epochs"],
        batch_size=model_config["batch_size"],
        verbose=1,
    )

    print("Evaluating...")
    mse = model.evaluate(
        x=[X_wide_test, X_deep_test],
        y=y_test,
        batch_size=model_config["batch_size"],
        verbose=1
    )
    print("Evaluation MSE:", mse)

    print("Saving ML model")
    model.save_weights(args.model_path)

if __name__ == "__main__":
    main()
