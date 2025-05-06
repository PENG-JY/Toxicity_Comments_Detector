import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
import logging
import argparse
from sklearn.metrics import precision_recall_fscore_support, roc_curve
import torch
from torch.utils.data import DataLoader
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from torch import nn
import torch.optim as optim



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SGDRegressionModel:
    """
    A class for training and predicting toxicity using the SGD Regressor model.

    Attributes:
        train_path (str): Path to the training dataset file.
        test_path (str): Path to the test dataset file.
        target_column (str): Name of the target column (toxicity score).
        ngram_range (tuple): n-gram range for text vectorization.
        max_features (int): Maximum number of features for the vectorizer.
        alpha (list): List of regularization strengths to tune.
        penalty (list): List of regularization types ('l1', 'l2').
    """
    def __init__(self, train_path: str, test_path: str, target_column: str = 'target', 
                 ngram_range=(1, 2), max_features=30000, alpha=None, penalty=None):
       
        self.train_path = train_path
        self.test_path = test_path
        self.target_column = target_column
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.alpha = alpha if alpha else [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        self.penalty = penalty if penalty else ['l1', 'l2']
        self.best_model = None
        self.best_error = float('inf')

    def preprocess_data(self):
        """
        Preprocess the data: load, clean, split into train/validation, and vectorize.

        Returns:
            tuple: Processed data for training, validation, and testing.
        """
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        train_df.dropna(inplace=True)
        test_df.dropna(inplace=True)

        # Extract features and targets
        feature = train_df[['comment_text']]
        output = train_df[[self.target_column]]

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            feature, output, test_size=0.25, random_state=5400
        )

        # Vectorize the text data
        cnt_vec = CountVectorizer(ngram_range=self.ngram_range, max_features=self.max_features)
        bow_train = cnt_vec.fit_transform(X_train['comment_text'])
        bow_cv = cnt_vec.transform(X_val['comment_text'])
        bow_test = cnt_vec.transform(test_df['comment_text'])  # Assuming test data has 'text' column

        return bow_train, bow_cv, y_train, y_val, bow_test, test_df, cnt_vec

    def hyperparameter_tuning(self, bow_train, y_train, bow_cv, y_val):
        """
        Tune hyperparameters for the SGD Regressor model.

        Parameters:
            bow_train: Bag-of-words features for training.
            y_train: Training targets.
            bow_cv: Bag-of-words features for validation.
            y_val: Validation targets.

        Returns:
            dict: Best model and related performance metrics.
        """
        logger.info("Starting hyperparameter tuning for SGD Regressor...")
        xticks = []
        tr_errors = []
        cv_errors = []

        for a in self.alpha:
            for p in self.penalty:
                xticks.append(f"Alpha-{a} Penalty-{p}")
                logger.info(f"Training with Alpha={a}, Penalty={p}")

                # Train SGD Regressor model
                model = SGDRegressor(alpha=a, penalty=p)
                model.fit(bow_train, y_train)

                # Calculate training error
                preds_train = model.predict(bow_train)
                err_train = mean_squared_error(y_train[self.target_column], preds_train)
                tr_errors.append(err_train)
                logger.info(f"Mean Squared Error on train set: {err_train}")

                # Calculate validation error
                preds_cv = model.predict(bow_cv)
                err_cv = mean_squared_error(y_val[self.target_column], preds_cv)
                cv_errors.append(err_cv)
                logger.info(f"Mean Squared Error on cv set: {err_cv}")

                if err_cv < self.best_error:
                    self.best_error = err_cv
                    self.best_model = model

        return {
            'best_model': self.best_model,
            'best_error': self.best_error,
            'tr_errors': tr_errors,
            'cv_errors': cv_errors,
            'xticks': xticks
        }

    def train_and_predict(self):
   
        # Load and preprocess data
        bow_train, bow_cv, y_train, y_val, bow_test, test_df, cnt_vec = self.preprocess_data()

        # Perform hyperparameter tuning to get the best model
        result = self.hyperparameter_tuning(bow_train, y_train, bow_cv, y_val)

        # Get the best model and make predictions on the test data
        best_model = result["best_model"]
        test_preds = best_model.predict(bow_test)

        # Combine test IDs, texts, and predictions
        predictions_df = pd.DataFrame({
            "id": test_df["id"],        # Test dataset's 'id'
            "comment_text": test_df["comment_text"],    # Test dataset's 'text'
            "Prediction": test_preds    # Predicted toxicity
        })

        return {"predictions": predictions_df}





class DecisionTreeModel:
    """
    A class for training and predicting toxicity using the Decision Tree Regressor.

    Attributes:
        train_path (str): Path to the training dataset file.
        test_path (str): Path to the test dataset file.
        target_column (str): Name of the target column (toxicity score).
        ngram_range (tuple): n-gram range for text vectorization.
        max_features (int): Maximum number of features for the vectorizer.
        max_depth (list): List of maximum depths for hyperparameter tuning.
        min_samples (list): List of minimum leaf sample sizes for tuning.
    """
  
    def __init__(self, train_path: str, test_path: str, target_column: str = 'target', 
                 ngram_range=(1, 2), max_features=30000, max_depth=[3, 5, 7], min_samples=[10, 100, 1000]):
       
        self.train_path = train_path
        self.test_path = test_path
        self.target_column = target_column
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples
        self.best_model = None
        self.best_error = float('inf')
    
    def preprocess_data(self,train_df,test_df):
        train_df['comment_text'] = train_df['comment_text'].fillna("").astype(str)
        test_df['comment_text'] = test_df['comment_text'].fillna("").astype(str)
        # Extract features and targets
        feature = train_df[['comment_text']]  
        output = train_df[[self.target_column]]
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(feature, output, test_size=0.25, random_state=5400)
        
        # Vectorize the text data
        cnt_vec = CountVectorizer(ngram_range=self.ngram_range, max_features=self.max_features)
        bow_train = cnt_vec.fit_transform(X_train['comment_text'])
        bow_cv = cnt_vec.transform(X_val['comment_text'])
        bow_test = cnt_vec.transform(test_df['comment_text'])  # Assuming test data has the same text column
        
        return bow_train, bow_cv, y_train, y_val, bow_test, test_df, cnt_vec
    
    def hyperparameter_tuning(self, bow_train, y_train, bow_cv, y_val):
        logger.info("Starting hyperparameter tuning for Decision Tree...")
        xticks = []
        tr_errors = []
        cv_errors = []
        self.best_model = None  
        self.best_error = float('inf')  

        for d in self.max_depth:
            for samp in self.min_samples_leaf:
                xticks.append(f"Depth-{d} Min Samples-{samp}")
                logger.debug(f"Training with Depth={d}, Min Samples={samp}")

                # Train Decision Tree model
                model = DecisionTreeRegressor(max_depth=d, min_samples_leaf=samp)
                model.fit(bow_train, y_train)

                # Calculate validation error
                preds_cv = model.predict(bow_cv)
                err_cv = mean_squared_error(y_val[self.target_column], preds_cv)
                cv_errors.append(err_cv)
                logger.info(f"Mean Squared Error on cv set: {err_cv}")

                # Update the best model
                if err_cv < self.best_error:
                    self.best_error = err_cv
                    self.best_model = model

        return {
            'best_model': self.best_model,
            'best_error': self.best_error,
            'tr_errors': tr_errors,
            'cv_errors': cv_errors,
            'xticks': xticks
        }



    
    def train_and_predict(self):
        # Read data
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        # Call preprocess_data with train_df and test_df
        bow_train, bow_cv, y_train, y_val, bow_test, test_df, cnt_vec = self.preprocess_data(train_df, test_df)

        # Perform hyperparameter tuning to get the best model
        result = self.hyperparameter_tuning(bow_train, y_train, bow_cv, y_val)

        # Get the best model and make predictions on the test data
        best_model = result["best_model"]
        test_preds = best_model.predict(bow_test)

        # Combine test IDs, texts, and predictions
        predictions_df = pd.DataFrame({
            "id": test_df["id"],        # Test dataset's 'id'
            "comment_text": test_df["comment_text"],  # Test dataset's 'text'
            "Prediction": test_preds    # Predicted toxicity
        })

        return {"predictions": predictions_df}




# Dataset class for PyTorch
class ToxicityDataset(torch.utils.data.Dataset):
    def __init__(self, texts, targets):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return {"texts": self.texts[index], "targets": self.targets[index]}


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

    def train_model(self, train_file, vocab_size, max_len, batch_size, epochs):
        train_data = pd.read_csv(train_file)
        train_data["comment_text"] = train_data["comment_text"].fillna("").astype(str)
        train_data["target"] = pd.to_numeric(train_data["target"], errors="coerce")
        train_data = train_data.dropna(subset=["target"])

        # Tokenizer and sequence processing
        tokenizer = Tokenizer(num_words=vocab_size)
        tokenizer.fit_on_texts(train_data["comment_text"])

        # Split train and validation data
        X_train, X_val, y_train, y_val = train_test_split(
            train_data["comment_text"], train_data["target"], test_size=0.25, random_state=5400
        )
        train_sequences = pad_sequences(
            tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding="post"
        )
        val_sequences = pad_sequences(
            tokenizer.texts_to_sequences(X_val), maxlen=max_len, padding="post"
        )

        # Prepare datasets
        train_dataset = ToxicityDataset(train_sequences, y_train.values)
        val_dataset = ToxicityDataset(val_sequences, y_val.values)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training configurations
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for batch in train_loader:
                texts, targets = batch["texts"].to(device), batch["targets"].to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = self(texts)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(train_loader):.4f}")

        return tokenizer

    def predict(self, test_file, tokenizer, max_len, batch_size):
        test_data = pd.read_csv(test_file)
        test_data["comment_text"] = test_data["comment_text"].fillna("")
        assert "id" in test_data.columns, "Test data must contain an 'id' column."
        assert "comment_text" in test_data.columns, "Test data must contain a 'comment_text' column."

        # Preprocess test data
        test_sequences = pad_sequences(
            tokenizer.texts_to_sequences(test_data["comment_text"]),
            maxlen=max_len,
            padding="post"
        )

        test_dataset = ToxicityDataset(test_sequences, [0] * len(test_sequences))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        self.to(device)

        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                texts = batch["texts"].to(device)
                outputs = self(texts)
                predictions.extend(outputs.cpu().numpy().flatten())

        return pd.DataFrame({
            "id": test_data["id"],
            "comment_text": test_data["comment_text"],
            "Prediction": predictions
        })

    def train_and_predict(self, train_file, test_file, vocab_size, max_len, batch_size, epochs):
        tokenizer = self.train_model(
            train_file=train_file,
            vocab_size=vocab_size,
            max_len=max_len,
            batch_size=batch_size,
            epochs=epochs
        )
        predictions = self.predict(
            test_file=test_file,
            tokenizer=tokenizer,
            max_len=max_len,
            batch_size=batch_size
        )
        return predictions, tokenizer



  


  

