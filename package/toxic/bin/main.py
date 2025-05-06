from package.toxic.utils.models import DecisionTreeModel, LSTMModel, SGDRegressionModel
from package.toxic.toxics.eval import evaluate_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse
import pandas as pd
import torch


def main():
    """
    Runs the toxic comment prediction process using different machine learning models.

    This function parses command-line arguments to select a model type, training data, 
    test data, and other parameters. Based on the selected model, it trains the model 
    using the training data, makes predictions on the test data, and outputs the results. 

    Arguments:
        - --train_file: Path to the training dataset file (CSV format).
        - --test_file: Path to the test dataset file (CSV format). This argument is required.
        - --model: Type of model to use for prediction. Options: "Tree", "SGD", "LSTM". This argument is required.
        - --output: Path to save the predictions. Default is "predictions.csv".
        - --print: Flag to print results to the console.
        - --evaluate: Flag to evaluate model performance on the training data, if ground truth is available.
    """
    parser = argparse.ArgumentParser(description="Run toxic comment prediction.")
    parser.add_argument("-train", "--train_file", type=str, default="package/toxic/bin/cleaned_train.csv",help="Path to the training dataset file (CSV format).")
    parser.add_argument("-test", "--test_file", type=str, required=True, help="Path to the test dataset file (CSV format).")
    parser.add_argument("-m", "--model", type=str, required=True, choices=["Tree", "SGD", "LSTM"], help="Model type.")
    parser.add_argument("-o", "--output", type=str, default="predictions.csv", help="Path to save predictions.")
    parser.add_argument("-p", "--print", action="store_true", help="Flag to print results.")
    parser.add_argument("-eval", "--evaluate", action="store_true", help="Evaluate model performance if ground truth is available.")
    args = parser.parse_args()

    # Initialize and train model
    if args.model == "Tree":
        model = DecisionTreeModel(train_path=args.train_file, test_path=args.test_file, target_column="target")
        result = model.train_and_predict()
        predictions_df = result["predictions"]
    elif args.model == "SGD":
        model = SGDRegressionModel(train_path=args.train_file, test_path=args.test_file, target_column="target")
        result = model.train_and_predict()
        predictions_df = result["predictions"]
    elif args.model == "LSTM":
        model = LSTMModel(vocab_size=10000, embedding_dim=64, hidden_dim=128)
        predictions_df, _ = model.train_and_predict(
            train_file=args.train_file,
            test_file=args.test_file,
            vocab_size=10000,
            max_len=100,
            batch_size=32,
            epochs=10
        )

    # Save predictions to a CSV file
    predictions_df.to_csv(args.output, index=False)

    # Optionally print predictions to the console
    if args.print:
        print(predictions_df)


        # Load datasets
    train_data = pd.read_csv(args.train_file)
    test_data = pd.read_csv(args.test_file)

    # Save predictions to a CSV file
    predictions_df.to_csv(args.output, index=False)

    # Optionally print predictions
    if args.print:
        print(predictions_df)

    if args.evaluate:
        print("\nEvaluating model performance on training data...")

    if args.model == "Tree":
        # Preprocess only the training data
        bow_train, _, y_train, _, _, _, _ = model.preprocess_data(train_df=train_data, test_df=test_data)

        # Check if the best model exists
        if model.best_model is None:
            print("Error: Decision Tree model was not trained correctly.")
            return

        # Predict on the training data
        train_predictions = model.best_model.predict(bow_train)

        # Flatten y_train and ensure lengths match
        y_train_flat = y_train["target"].values.flatten()
        if len(y_train_flat) == len(train_predictions):
            metrics = evaluate_model(y_train_flat, train_predictions)
            print("\nTraining Data Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        else:
            print(f"Error: Length mismatch (y_true: {len(y_train_flat)}, y_pred: {len(train_predictions)})")

    elif args.model == "SGD":
        # Preprocess only the training data
        bow_train, _, y_train, _, _, _, _ = model.preprocess_data()

        # Check if the best model exists
        if model.best_model is None:
            print("Error: SGD model was not trained correctly.")
            return

        # Predict on the training data
        train_predictions = model.best_model.predict(bow_train)

        # Flatten y_train and ensure lengths match
        y_train_flat = y_train["target"].values.flatten()
        if len(y_train_flat) == len(train_predictions):
            metrics = evaluate_model(y_train_flat, train_predictions)
            print("\nTraining Data Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        else:
            print(f"Error: Length mismatch (y_true: {len(y_train_flat)}, y_pred: {len(train_predictions)})")

    elif args.model == "LSTM":
        # Train tokenizer and predict on training data
        train_data["comment_text"] = train_data["comment_text"].fillna("").astype(str)
        tokenizer = model.train_model(
            train_file=args.train_file, vocab_size=10000, max_len=100, batch_size=32, epochs=1
        )
        train_sequences = pad_sequences(
            tokenizer.texts_to_sequences(train_data["comment_text"]), maxlen=100, padding="post"
        )
        train_predictions = model(torch.tensor(train_sequences, dtype=torch.long)).detach().numpy()

        # Evaluate
        y_train = train_data["target"].values
        if len(y_train) == len(train_predictions):
            metrics = evaluate_model(y_train, train_predictions)
            print("\nTraining Data Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        else:
            print(f"Error: Length mismatch (y_true: {len(y_train)}, y_pred: {len(train_predictions)})")






if __name__ == "__main__":
    main()




