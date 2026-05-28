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
    'n_estimators': 1000,       # Số lượng cây quyết định
    'max_depth': 3,             # Độ sâu tối đa của mỗi cây
    'max_features': 60,         # Lấy ngẫu nhiên 60 biến (trong số 88+ biến) tại mỗi lần phân nhánh
    'learning_rate': 0.1,       # Tốc độ học
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'loss': 'squared_error',
    'random_state': RANDOM_STATE
}

# Thư viện 2: XGBoost Regressor
XGB_PARAMS = {
    'max_depth': 3,
    'learning_rate': 0.01,
    'n_estimators': 5000,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'objective': 'reg:squarederror',
    'tree_method': 'hist', # Tăng tốc bằng histogram
    'random_state': RANDOM_STATE,
    'n_jobs': -1           # Dùng toàn bộ nhân CPU
}

# Thư viện 3: LightGBM Regressor
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': RANDOM_STATE,
    'learning_rate': 0.02,      # Tốc độ học nhỏ
    'n_estimators': 3000,       # Số cây lớn để early stopping hoạt động
    'max_depth': 8,             # Theo cấu hình Jinkai Tan
    'num_leaves': 7,            # Nhỏ để tránh overfitting
    'feature_fraction': 0.6,    # Lấy 70% features mỗi lần fit cây
    'lambda_l1': 0.1,           # Regularization
    'lambda_l2': 1,
}
