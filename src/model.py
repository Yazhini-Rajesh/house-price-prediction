import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def encode_categoricals(df):
    df = df.copy()
    for col in df.select_dtypes('object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def compare_models(X_train, y_train):
    models = {
        'Ridge Regression':  Ridge(alpha=10),
        'Random Forest':     RandomForestRegressor(n_estimators=200, random_state=42),
        'XGBoost':           XGBRegressor(n_estimators=1000, learning_rate=0.05,
                                          max_depth=4, random_state=42, verbosity=0),
    }
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train,
                                 cv=5, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores.mean())
        print(f"{name:20s}  RMSE: {rmse:.4f}")
        results[name] = rmse

    best_name = min(results, key=results.get)
    print(f"\nBest model: {best_name}")
    models[best_name].fit(X_train, y_train)
    return models[best_name]