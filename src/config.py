import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FINAL_DATA_DIR = os.path.join(BASE_DIR, "data", "final")

FINAL_FEATURES_PATH = os.path.join(FINAL_DATA_DIR, "features.csv")
TARGET_PATH = os.path.join(FINAL_DATA_DIR, "target.csv")

RESULTS_DIR = os.path.join(BASE_DIR, "experiments")

RANDOM_STATE = 42


MERGE_LABELS = ['SID', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'LAT', 'LON']

DROPS_COLUMNS = ['SID', 'YEAR', 'MONTH', 'DAY', 'HOUR'] + [
                col for col in [f'DELTA_LAT_{i}h' for i in [6, 12, 18, 24]]] + [
                col for col in [f'DELTA_LON_{i}h' for i in [6, 12, 18, 24]]]


HORIZONS = [6, 12, 24]

# Thư viện 1: Scikit-Learn Gradient Boosting Decision Tree (GBDT)
GBDT_PARAMS = {
    'n_estimators': 800,           # Fewer iterations; no early stopping available
    'max_depth': 2,                # Very shallow - limit memorization of chain noise
    'max_features': 30,            # Fewer features per split - avoid fitting to chain noise
    'learning_rate': 0.03,         # Slower learning for stability across links
    'subsample': 0.6,              # Strong subsample - reduce overfitting to chain features
    'min_samples_split': 10,       # Stricter split requirement
    'min_samples_leaf': 8,         # Larger leaves - more robust predictions
    'loss': 'squared_error',
    'random_state': RANDOM_STATE,
}

# Thư viện 2: XGBoost Regressor
XGB_PARAMS = {
    'max_depth': 3,
    'learning_rate': 0.01,
    'n_estimators': 2500,       # 5000 quá rủi ro cho Chains nếu không early stop gắt
    'subsample': 0.75,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,           # QUAN TRỌNG (L1): Triệt tiêu các nhiễu nhỏ
    'reg_lambda': 3.0,          # QUAN TRỌNG (L2): Giảm trọng số của các biến dự đoán từ mốc trước
    'gamma': 0.1,               # Yêu cầu loss giảm tối thiểu mới cho tỉa nhánh
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Thư viện 3: LightGBM Regressor
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': RANDOM_STATE,
    'learning_rate': 0.01,         # Giảm từ 0.015
    'n_estimators': 1500,          # Giảm; early stopping sẽ dừng sớm
    'max_depth': 3,                # Giảm từ 4
    'num_leaves': 8,               # Giảm mạnh từ 15, phù hợp max_depth=3
    'feature_fraction': 0.5,       # Giảm từ 0.6
    'bagging_fraction': 0.65,      # Giảm từ 0.75
    'bagging_freq': 1,
    'lambda_l1': 1.5,              # Tăng từ 0.5
    'lambda_l2': 8.0,              # Tăng từ 5.0 -> quan trọng để giới hạn propagation
    'min_child_samples': 30,       # Tăng từ 20 -> lá phải tập trung hơn
}


