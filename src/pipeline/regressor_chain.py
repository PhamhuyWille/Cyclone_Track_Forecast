import lightgbm as lgb
import config as config
import models.model as model


class ChainForecastingStrategy:

    def __init__(self, model_name: str, order_type: str = 'horizon_first'):
        """order_type:

        - 'horizon_first': Cặp [Lat, Lon] gối đầu theo thời gian (Hiện tại)
        - 'coordinate_first_lat': Luyện hết Lat 6h-24h rồi mới luyện Lon 6h-24h
        - 'strict_interleaved': Chạy tuần tự Lat6h -> Lon6h -> Lat12h -> Lon12h...
        """
        self.model_name = model_name.lower()
        self.order_type = order_type

        self.model_lat = {
            h: model.get_model(model_name) for h in config.HORIZONS
        }
        self.model_lon = {
            h: model.get_model(model_name) for h in config.HORIZONS
        }

    def _fit_model(self, model_instance, X_tr, y_tr, X_v, y_v):
        """Hàm bọc xử lý fit tự động ngắt cho các dòng model"""
        if self.model_name == 'lightgbm':
            model_instance.fit(
                X_tr,
                y_tr,
                eval_set=[(X_v, y_v)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        elif self.model_name == 'xgboost':
            model_instance.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
        elif self.model_name == 'gdbt':
            model_instance.fit(X_tr, y_tr)

    def train(
        self,
        X_train,
        y_train_lat_dict,
        y_train_lon_dict,
        X_val,
        y_val_lat_dict,
        y_val_lon_dict,
    ):
        current_X_train = X_train.copy()
        current_X_val = X_val.copy()

        # ======================================================================
        # TRÌNH TỰ 1: HORIZON FIRST (SỬA TRIỆT ĐỂ LỖI 84 VS 83 FEATURES)
        # ======================================================================
        if self.order_type == 'horizon_first':
            for h in config.HORIZONS:
                # Huấn luyện song song trên cùng một không gian đặc trưng nền tảng mốc h
                self._fit_model(
                    self.model_lat[h],
                    current_X_train,
                    y_train_lat_dict[h],
                    current_X_val,
                    y_val_lat_dict[h],
                )
                self._fit_model(
                    self.model_lon[h],
                    current_X_train,
                    y_train_lon_dict[h],
                    current_X_val,
                    y_val_lon_dict[h],
                )

                # FIX LỖI: Trích xuất toàn bộ predict ra biến tạm trước khi thay đổi hình dáng dataframe
                pred_lat_tr = self.model_lat[h].predict(current_X_train)
                pred_lon_tr = self.model_lon[h].predict(current_X_train)
                pred_lat_v = self.model_lat[h].predict(current_X_val)
                pred_lon_v = self.model_lon[h].predict(current_X_val)

                # Sau khi predict đồng pha xong mới tiến hành gập đặc trưng gối đầu chuỗi
                current_X_train[f'chain_lat_{h}h'] = pred_lat_tr
                current_X_train[f'chain_lon_{h}h'] = pred_lon_tr
                current_X_val[f'chain_lat_{h}h'] = pred_lat_v
                current_X_val[f'chain_lon_{h}h'] = pred_lon_v

            # ======================================================================
            # TRÌNH TỰ 2: COORDINATE FIRST (LAT -> LON)
            # ======================================================================
        elif self.order_type == 'coordinate_first_lat':
        # Bước A: Huấn luyện và đóng gói toàn bộ các mốc Vĩ độ trước
            for h in config.HORIZONS:
                self._fit_model(
                    self.model_lat[h],
                    current_X_train,
                    y_train_lat_dict[h],
                    current_X_val,
                    y_val_lat_dict[h],
                )
                current_X_train[f'chain_lat_{h}h'] = self.model_lat[h].predict(
                    current_X_train
                )
                current_X_val[f'chain_lat_{h}h'] = self.model_lat[h].predict(
                    current_X_val
                )

            # Bước B: Huấn luyện các mốc Kinh độ (Đã được sửa lại y_val_lon_dict chính xác)
            for h in config.HORIZONS:
                self._fit_model(
                    self.model_lon[h],
                    current_X_train,
                    y_train_lon_dict[h],
                    current_X_val,
                    y_val_lon_dict[h],
                )
                current_X_train[f'chain_lon_{h}h'] = self.model_lon[h].predict(
                    current_X_train
                )
                current_X_val[f'chain_lon_{h}h'] = self.model_lon[h].predict(
                    current_X_val
                )

        # ======================================================================
        # TRÌNH TỰ 3: STRICT INTERLEAVED (ZIC-ZAC CHẶT CHẼ)
        # ======================================================================
        elif self.order_type == 'strict_interleaved':
            for h in config.HORIZONS:
                # 1. Cạnh xích Lat mốc h nhìn về quá khứ kết hợp mốc trước đó
                self._fit_model(
                    self.model_lat[h],
                    current_X_train,
                    y_train_lat_dict[h],
                    current_X_val,
                    y_val_lat_dict[h],
                )
                current_X_train[f'chain_lat_{h}h'] = self.model_lat[h].predict(
                    current_X_train
                )
                current_X_val[f'chain_lat_{h}h'] = self.model_lat[h].predict(
                    current_X_val
                )

                # 2. Cạnh xích Lon mốc h thừa kế ngay lập tức Lat của chính mốc h vừa sinh
                self._fit_model(
                    self.model_lon[h],
                    current_X_train,
                    y_train_lon_dict[h],
                    current_X_val,
                    y_val_lon_dict[h],
                )
                current_X_train[f'chain_lon_{h}h'] = self.model_lon[h].predict(
                    current_X_train
                )
                current_X_val[f'chain_lon_{h}h'] = self.model_lon[h].predict(
                    current_X_val
                )

    def predict(self, X_test):
        current_X = X_test.copy()
        predictions_lat = {}
        predictions_lon = {}

        if self.order_type == 'horizon_first':
            for h in config.HORIZONS:
                predictions_lat[h] = self.model_lat[h].predict(current_X)
                predictions_lon[h] = self.model_lon[h].predict(current_X)
                current_X[f'chain_lat_{h}h'] = predictions_lat[h]
                current_X[f'chain_lon_{h}h'] = predictions_lon[h]

        elif self.order_type == 'coordinate_first_lat':
            for h in config.HORIZONS:
                predictions_lat[h] = self.model_lat[h].predict(current_X)
                current_X[f'chain_lat_{h}h'] = predictions_lat[h]
            for h in config.HORIZONS:
                predictions_lon[h] = self.model_lon[h].predict(current_X)
                current_X[f'chain_lon_{h}h'] = predictions_lon[h]

        elif self.order_type == 'strict_interleaved':
            for h in config.HORIZONS:
                predictions_lat[h] = self.model_lat[h].predict(current_X)
                current_X[f'chain_lat_{h}h'] = predictions_lat[h]

                predictions_lon[h] = self.model_lon[h].predict(current_X)
                current_X[f'chain_lon_{h}h'] = predictions_lon[h]

        return predictions_lat, predictions_lon