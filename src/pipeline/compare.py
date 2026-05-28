import pandas as pd
import numpy as np
import os
import config as config
from pipeline.normal import IndependentForecastingStrategy
from pipeline.regressor_chain import ChainForecastingStrategy
from pipeline.cliper import CLIPERForecastingStrategy
from evaluation.metrics import haversine_distance, evaluate_advanced_metrics

def load_and_split_data():
    """Đọc dữ liệu và chia thành 3 phần Train/Valid/Test cho tất cả các mốc"""
    features = pd.read_csv(config.FINAL_FEATURES_PATH)
    target = pd.read_csv(config.TARGET_PATH)

    df = features.merge(target, on=config.MERGE_LABELS, how='left')

    # Loại bỏ NaN trên toàn bộ các cột Target để tập dữ liệu đồng nhất
    all_targets = []
    for h in config.HORIZONS:
        all_targets.extend([f'DELTA_LAT_{h}h', f'DELTA_LON_{h}h'])
    df = df.dropna(subset=all_targets)

    # Chia tập dữ liệu theo năm như logic của bạn
    train_df = df[(df['YEAR'] >= 2000) & (df['YEAR'] <= 2020)].copy()
    val_df = df[(df['YEAR'] >= 2021) & (df['YEAR'] <= 2022)].copy()  
    test_df = df[(df['YEAR'] >= 2023) & (df['YEAR'] <= 2024)].copy()

    # Gom tách target thành các dictionary tương ứng với từng mốc thời gian
    y_train_lat, y_train_lon = {}, {}
    y_val_lat, y_val_lon = {}, {}
    y_test_lat, y_test_lon = {}, {}

    for h in config.HORIZONS:
        y_train_lat[h] = train_df[f'DELTA_LAT_{h}h']
        y_train_lon[h] = train_df[f'DELTA_LON_{h}h']
        y_val_lat[h] = val_df[f'DELTA_LAT_{h}h']
        y_val_lon[h] = val_df[f'DELTA_LON_{h}h']
        y_test_lat[h] = test_df[f'DELTA_LAT_{h}h']
        y_test_lon[h] = test_df[f'DELTA_LON_{h}h']

    X_train = train_df.drop(columns=config.DROPS_COLUMNS)
    X_val = val_df.drop(columns=config.DROPS_COLUMNS)
    X_test = test_df.drop(columns=config.DROPS_COLUMNS)

    return X_train, X_val, X_test, y_train_lat, y_train_lon, y_val_lat, y_val_lon, y_test_lat, y_test_lon, test_df


def run_experiment_matrix():
    # Nạp ma trận dữ liệu tập trung hoàn chỉnh
    (X_train, X_val, X_test, 
     y_train_lat, y_train_lon, 
     y_val_lat, y_val_lon, 
     y_test_lat, y_test_lon, test_df) = load_and_split_data()
    
    report_rows = []
    
    print("\n==================== THÍ NGHIỆM 0: CLIPER BASELINE ====================")
    cliper_runner = CLIPERForecastingStrategy()
    cliper_runner.train(X_train, y_train_lat, y_train_lon) # Chạy train tuyến tính nhanh chớp mắt
    pred_clip_lat, pred_clip_lon = cliper_runner.predict(X_test)

    for h in config.HORIZONS:
        true_lat = test_df['LAT'].values + y_test_lat[h].values
        true_lon = test_df['LON'].values + y_test_lon[h].values
        
        pred_clip_lat_geo = test_df['LAT'].values + pred_clip_lat[h]
        pred_clip_lon_geo = test_df['LON'].values + pred_clip_lon[h]
        
        err_clip = haversine_distance(true_lat, true_lon, pred_clip_lat_geo, pred_clip_lon_geo)
        err_clip_mean = np.mean(err_clip)
        print(f"   Mốc +{h}h -> Sai số CLIPER: {err_clip_mean:.2f} km")
        
        v_past = test_df[f'V_Actual_{h}h'].values

        actual = np.column_stack((y_test_lat[h], y_test_lon[h]))
        y_test_actual = pd.DataFrame(actual)

        predict = np.column_stack((pred_clip_lat[h], pred_clip_lon[h]))
        y_test_predicted = pd.DataFrame(predict)

        directional_stability, velocity_sensitivity = evaluate_advanced_metrics(y_test_actual, y_test_predicted, v_past)

        # Lưu kết quả CLIPER vào bảng báo cáo
        report_rows.append({
            "Model/Strategy": "CLIPER_BASELINE",
            "Horizon": f"+{h}h",
            "Error_KM": round(err_clip_mean, 2),
            'Directional_Stability': directional_stability,
            'Moving_Velocity_Sensitivity': velocity_sensitivity
        })
    

    models = ['lightgbm', 'xgboost', 'gdbt'] 
    
    for model_name in models:
        print(f"\n==================== THÍ NGHIỆM MÔ HÌNH: {model_name.upper()} ====================")
        
        # 1. Huấn luyện & Dự báo kỹ thuật Independent
        normal = IndependentForecastingStrategy(model_name)
        normal.train(X_train, y_train_lat, y_train_lon, X_val, y_val_lat, y_val_lon)
        pred_ind_lat, pred_ind_lon = normal.predict(X_test)

        for h in config.HORIZONS:
            # Tọa độ thực tế đích (LAT, LON gốc + DELTA thực tế)
            true_lat = test_df['LAT'].values + y_test_lat[h].values
            true_lon = test_df['LON'].values + y_test_lon[h].values
            
            # Tọa độ dự báo đích Independent
            pred_ind_lat_geo = test_df['LAT'].values + pred_ind_lat[h]
            pred_ind_lon_geo = test_df['LON'].values + pred_ind_lon[h]
        
            # Tính toán sai số km thực tế
            err_ind = haversine_distance(true_lat, true_lon, pred_ind_lat_geo, pred_ind_lon_geo)
            err_ind_mean = np.mean(err_ind)

            print(f"Mốc +{h}h -> Sai số Independent: {err_ind_mean:.2f} km")

            v_past = test_df[f'V_Actual_{h}h'].values
            y_test_actual = pd.DataFrame(np.column_stack((y_test_lat[h], y_test_lon[h])))
            y_test_predicted = pd.DataFrame(np.column_stack((pred_ind_lat[h], pred_ind_lon[h])))

            directional_stability, velocity_sensitivity = evaluate_advanced_metrics(y_test_actual, y_test_predicted, v_past)

            report_rows.append({
                "Model/Strategy": model_name.upper() + '_NORMAL',
                "Horizon": f"+{h}h",
                "Error_KM": round(err_ind_mean, 2),
                'Directional_Stability': directional_stability,
                'Moving_Velocity_Sensitivity': velocity_sensitivity
            })

        
        # 2. Huấn luyện & Dự báo kỹ thuật Regressor Chain
        chain_orderings = ['horizon_first', 'coordinate_first_lat', 'strict_interleaved']
        
        for order in chain_orderings:
            print(f"-> Đang chạy chiến lược Regressor Chain [Trình tự: {order.upper()}]...")
            chain = ChainForecastingStrategy(model_name=model_name, order_type=order)
            chain.train(X_train, y_train_lat, y_train_lon, X_val, y_val_lat, y_val_lon)
            pred_chain_lat, pred_chain_lon = chain.predict(X_test)

            for h in config.HORIZONS:
                # Tọa độ thực tế đích (LAT, LON gốc + DELTA thực tế)
                true_lat = test_df['LAT'].values + y_test_lat[h].values
                true_lon = test_df['LON'].values + y_test_lon[h].values
                
                # Tọa độ dự báo đích Regressor Chain
                pred_chain_lat_geo = test_df['LAT'].values + pred_chain_lat[h]
                pred_chain_lon_geo = test_df['LON'].values + pred_chain_lon[h]

                err_chain = haversine_distance(true_lat, true_lon, pred_chain_lat_geo, pred_chain_lon_geo)
                err_chain_mean = np.mean(err_chain)
                
                print(f"Mốc +{h}h -> Sai số Regressor Chain: {err_chain_mean:.2f} km")
                
                v_past = test_df[f'V_Actual_{h}h'].values
                y_test_actual = pd.DataFrame(np.column_stack((y_test_lat[h], y_test_lon[h])))
                y_test_chain = pd.DataFrame(np.column_stack((pred_chain_lat[h], pred_chain_lon[h])))

                ds_chain, vs_chain = evaluate_advanced_metrics(y_test_actual, y_test_chain, v_past)

                
                report_rows.append({
                    "Model/Strategy": f"{model_name.upper()}_CHAIN_{order.upper()}",
                    "Horizon": f"+{h}h",
                    "Error_KM": round(err_chain_mean, 2),
                    'Directional_Stability': ds_chain,
                    'Moving_Velocity_Sensitivity': vs_chain
                })
        
    # Xuất file báo cáo kết quả chi tiết
    df_report = pd.DataFrame(report_rows)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(config.RESULTS_DIR, "final_comparison_report_ver02.csv")
    df_report.to_csv(report_path, index=False)
    print(f"\n[THÀNH CÔNG] Đã lưu bảng thống kê đối chiếu cấu trúc đồ án tại: {report_path}")

if __name__ == "__main__":
    run_experiment_matrix()
