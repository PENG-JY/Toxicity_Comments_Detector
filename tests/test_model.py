import pytest
import warnings
import pandas as pd
from package.toxic.utils.parse_csv import DecisionTreeModel, SGDRegressionModel, LSTMModel

# Ignore specific warning types
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Path to the test data
TEST_DATA_PATH = "tests/test_data_v2.csv"

# Fixture to load test data
@pytest.fixture
def load_test_data():
    """
    Load test data from the specified CSV file.
    Returns:
        DataFrame: The loaded test data.
    """
    return pd.read_csv(TEST_DATA_PATH)

# Test Decision Tree Model
def test_decision_tree(load_test_data):
    """
    Test the Decision Tree model:
    - Train the model on the test dataset.
    - Predict values and print predictions alongside true values.
    """
    df = load_test_data  # Load the test data

    # Initialize and train the Decision Tree model
    model = DecisionTreeModel(train_path=TEST_DATA_PATH, test_path=TEST_DATA_PATH, target_column="target")
    result = model.train_and_predict()

    # Extract predictions and true values
    predictions = result["predictions"]["Prediction"].values
    y_true = df["target"].values.astype(float).ravel()  # Flatten to a 1D array

    # Print results
    print("\nDecision Tree Model:")
    print(f"Predictions: {predictions}")
    print(f"True Values: {y_true}")

    # Validate the prediction and target length
    assert len(predictions) == len(y_true), "Prediction length does not match True values."

# Test SGD Regression Model
def test_sgd_regression(load_test_data):
    """
    Test the SGD Regression model:
    - Train the model on the test dataset.
    - Predict values and print predictions alongside true values.
    """
    df = load_test_data  # Load the test data

    # Initialize and train the SGD Regression model
    model = SGDRegressionModel(train_path=TEST_DATA_PATH, test_path=TEST_DATA_PATH, target_column="target")
    result = model.train_and_predict()

    # Extract predictions and true values
    predictions = result["predictions"]["Prediction"].values
    y_true = df["target"].values.astype(float).ravel()  # Flatten to a 1D array

    # Print results
    print("\nSGD Regression Model:")
    print(f"Predictions: {predictions}")
    print(f"True Values: {y_true}")

    # Validate the prediction and target length
    assert len(predictions) == len(y_true), "Prediction length does not match True values."

# Test LSTM Model
def test_lstm_model(load_test_data):
    """
    Test the LSTM model:
    - Train the model on the test dataset.
    - Predict values and print predictions alongside true values.
    """
    df = load_test_data  # Load the test data

    # Initialize and train the LSTM model
    model = LSTMModel(vocab_size=10000, embedding_dim=64, hidden_dim=128)
    predictions_df, _ = model.train_and_predict(
        train_file=TEST_DATA_PATH,
        test_file=TEST_DATA_PATH,
        vocab_size=10000,
        max_len=100,
        batch_size=32,
        epochs=1
    )

    # Extract predictions and true values
    predictions = predictions_df["Prediction"].values
    y_true = df["target"].values.astype(float).ravel()  # Flatten to a 1D array

    # Print results
    print("\nLSTM Model:")
    print(f"Predictions: {predictions}")
    print(f"True Values: {y_true}")

    # Validate the prediction and target length
    assert len(predictions) == len(y_true), "Prediction length does not match True values."









    