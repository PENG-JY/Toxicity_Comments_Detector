from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model predictions using common regression metrics.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values by the model.

    Returns:
        dict: Evaluation metrics (MSE, RMSE, MAE, R²).
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    }

