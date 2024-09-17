from preprocessing.preprocessing import load_and_resample_data, create_time_features
from models.model_trainer import train_linear_regression, train_lightgbm, train_xgboost
from evaluation.evaluation import calculate_rmse
from visualizations.plotter import plot_orders_per_hour, plot_orders_per_day
from sklearn.model_selection import train_test_split

# 1. Cargar y preprocesar los datos
data_resampled = load_and_resample_data('datasets/taxi.csv')
data_resampled = create_time_features(data_resampled)

# 2. División de los datos
X = data_resampled[['year', 'month', 'day', 'hour', 'dayofweek']]
y = data_resampled['num_orders'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

# 3. Entrenamiento de modelos
model_lr = train_linear_regression(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
rmse_lr = calculate_rmse(y_test, y_pred_lr)
print(f'RMSE Regresión Lineal: {rmse_lr}')

# Entrenamiento y evaluación de LightGBM
param_grid_lgb = {'num_leaves': [31, 50], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
model_lgb = train_lightgbm(X_train, y_train, param_grid_lgb)
y_pred_lgb = model_lgb.predict(X_test)
rmse_lgb = calculate_rmse(y_test, y_pred_lgb)
print(f'RMSE LightGBM: {rmse_lgb}')

# Entrenamiento y evaluación de XGBoost
param_grid_xgb = {'max_depth': [3, 6], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
model_xgb = train_xgboost(X_train, y_train, param_grid_xgb)
y_pred_xgb = model_xgb.predict(X_test)
rmse_xgb = calculate_rmse(y_test, y_pred_xgb)
print(f'RMSE XGBoost: {rmse_xgb}')

# 4. Visualización de resultados
specific_day = '2018-03-01'

plot_orders_per_hour(day=specific_day, data_resampled=data_resampled)

start_date = '2018-03-01'
end_date = '2018-03-31'

plot_orders_per_day(s_date=start_date, e_date=end_date, data_resampled=data_resampled)
