from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(y_true, y_pred, model_name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    return rmse, mae

def compare_models(dt_scores, xgb_scores):
    dt_rmse, dt_mae = dt_scores
    xgb_rmse, xgb_mae = xgb_scores

    print("\nModel Comparison:")
    if xgb_rmse < dt_rmse:
        print("XGBoost performed better based on RMSE.")
    else:
        print("Decision Tree performed better based on RMSE.")

    return {
        "Decision Tree": {"RMSE": dt_rmse, "MAE": dt_mae},
        "XGBoost": {"RMSE": xgb_rmse, "MAE": xgb_mae}
    }
