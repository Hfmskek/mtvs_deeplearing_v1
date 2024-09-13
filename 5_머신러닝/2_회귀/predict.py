import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 데이터 로드 및 전처리
bike_df = pd.read_csv('bike_train.csv')
bike_df['datetime'] = pd.to_datetime(bike_df['datetime'])

# 특성 엔지니어링
bike_df['year'] = bike_df.datetime.dt.year
bike_df['month'] = bike_df.datetime.dt.month
bike_df['day'] = bike_df.datetime.dt.day
bike_df['hour'] = bike_df.datetime.dt.hour
bike_df['dayofweek'] = bike_df.datetime.dt.dayofweek
bike_df['is_weekend'] = bike_df.dayofweek.isin([5, 6]).astype(int)
bike_df['season'] = bike_df.datetime.dt.month % 12 // 3 + 1

# 상호작용 특성 추가
bike_df['temp_atemp'] = bike_df['temp'] * bike_df['atemp']
bike_df['humidity_windspeed'] = bike_df['humidity'] * bike_df['windspeed']

# 이상치 처리
def clip_outliers(df, column, lower_percentile=1, upper_percentile=99):
    lower = np.percentile(df[column], lower_percentile)
    upper = np.percentile(df[column], upper_percentile)
    df[column] = df[column].clip(lower, upper)
    return df

bike_df = clip_outliers(bike_df, 'count')

# 데이터 분할
X = bike_df.drop(['count', 'datetime', 'casual', 'registered'], axis=1)
y = bike_df['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 정의
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42)
}

# 하이퍼파라미터 그리드
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'num_leaves': [31, 63, 127]
    }
}

# 그리드 서치 및 모델 평가
best_rmse = float('inf')
best_model = None

for name, model in models.items():
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name],
                               cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    y_pred = grid_search.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"{name} - Best Parameters: {grid_search.best_params_}")
    print(f"{name} - Test RMSE: {rmse}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = grid_search.best_estimator_

print(f"\nBest Overall Model: {type(best_model).__name__}")
print(f"Best Overall RMSE: {best_rmse}")