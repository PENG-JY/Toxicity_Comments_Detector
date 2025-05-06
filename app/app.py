import os
import pandas as pd
from flask import Flask, request, jsonify
from package.toxic.utils.models import DecisionTreeModel, LSTMModel, SGDRegressionModel
from package.toxic.toxics.eval import evaluate_model
import torch
from flask_cors import CORS  # Import CORS library
import traceback

# Initialize Flask application
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts the toxicity of a comment based on the selected model (Decision Tree, SGD, or LSTM).

    This function receives a JSON request containing the user ID, comment text, and model type. 
    It validates the input data, selects the appropriate machine learning model, performs the prediction, 
    and returns the predicted toxicity score.

    Steps:
    1. Parse the input data (user ID, comment text, and model type).
    2. Validate input data for required fields and correct formats.
    3. Use the specified model type (Decision Tree, SGD, or LSTM) to make a prediction.
    4. Return the prediction result in JSON format.
    """
    test_file_path = "temp_test_data.csv"  # Define the path for the test file
    try:
        input_data = request.get_json()

        user_id = input_data.get('id')
        comment_text = input_data.get('comment_text')
        model_type = input_data.get('model')

        # Validate input data
        if not user_id or not isinstance(user_id, str) or user_id.strip() == "":
            return jsonify({"error": "Invalid or missing 'id'. It must be a non-empty string."}), 400
        if not comment_text or not isinstance(comment_text, str) or comment_text.strip() == "":
            return jsonify({"error": "Invalid or missing 'comment_text'. It must be a non-empty string."}), 400
        if model_type not in ["Tree", "SGD", "LSTM"]:
            return jsonify({"error": "Invalid 'model'. Valid options are: 'Tree', 'SGD', 'LSTM'."}), 400

        # Process model prediction
        test_df = pd.DataFrame({"id": [user_id], "comment_text": [comment_text]})

        test_df.to_csv(test_file_path, index=False)  # Save as a temporary file

        # Model prediction logic
        if model_type == "Tree":
            model = DecisionTreeModel(train_path="../data/data-clean/cleaned_train.csv", test_path=test_file_path, target_column="target")
            result = model.train_and_predict()
            predictions_df = result["predictions"]
            predictions = predictions_df["Prediction"].values  # Get the predicted value
        elif model_type == "SGD":
            model = SGDRegressionModel(train_path="../data/data-clean/cleaned_train.csv", test_path=test_file_path, target_column="target")
            result = model.train_and_predict()
            predictions_df = result["predictions"]
            predictions = predictions_df["Prediction"].values  # Get the predicted value
        elif model_type == "LSTM":
            model = LSTMModel(vocab_size=10000, embedding_dim=64, hidden_dim=128)
            predictions_df, _ = model.train_and_predict(
                train_file="../data/data-clean/cleaned_train.csv",
                test_file=test_file_path,
                vocab_size=10000,
                max_len=100,
                batch_size=32,
                epochs=10
            )
            # Modify the returned column name to 'Prediction'
            if "Prediction" not in predictions_df.columns:
                predictions_df = predictions_df.rename(columns={"toxicity_score": "Prediction"})
            predictions = predictions_df["Prediction"].values  # Get the predicted value
        else:
            return jsonify({"error": "Unexpected model type."}), 400

        # Return the predicted result as an array (instead of DataFrame)
        return jsonify([{"Prediction": predictions[0]}])  # Convert to list and return

    except Exception as e:
        # Catch exceptions and print detailed error messages
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    finally:
        # Delete temporary file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            print(f"Temporary file {test_file_path} has been deleted.")

if __name__ == '__main__':
    app.run(debug=True)
