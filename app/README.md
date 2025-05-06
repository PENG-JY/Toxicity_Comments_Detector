# Toxicity Analysis Flask App

This is a simple Flask application that allows users to analyze the toxicity of text comments using machine learning models. The app provides an API endpoint to receive text comments, predict toxicity scores using various models, and return the result.

## Features
- Supports three different models for toxicity prediction:
  - **Decision Tree (Tree)**
  - **SGD Regression (SGD)**
  - **LSTM (LSTM)**
- Provides an easy-to-use web interface to input data and get predictions.

## Installation

1. **Clone the repository**:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2. **Install dependencies**:
    Make sure you have Python 3.7+ installed. You can install the required dependencies using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

    If you don't have `requirements.txt`, manually install Flask and any other necessary dependencies:
    ```bash
    pip install flask pandas scikit-learn torch
    ```

3. **Set up the model package**:
    Ensure the model package is installed. If not, run:
    ```bash
    pip install .
    ```

## Usage

### Running the Flask App

To run the Flask application, execute the following in your terminal:

```bash
python app.py
```

You can interact with the API by sending POST requests to the `/predict` endpoint.

### API Endpoint

- **POST /predict**:
    - **Request Body**: JSON object with the following fields:
        - `id`: User ID (String)
        - `comment_text`: The text comment to analyze (String)
        - `model`: The model to use for prediction. Options: `Tree`, `SGD`, `LSTM`
    - **Response**: JSON object with the predicted toxicity score.
        ```json
        {
            "Prediction": 0.123456
        }
        ```


### Web Interface

You can also interact with the app via the provided web interface (`index.html`). The interface allows you to input a comment and choose the model, then it will display the toxicity score.

Simply open the `index.html` file in a browser and input your data. Click "Analyze Toxicity" to receive the prediction.

## Directory Structure

```
/project-directory
    /app
        - app.py               # Flask application file
    /static                    # Static files like CSS, JS
    /templates                 # HTML templates
    - requirements.txt         # Project dependencies
```