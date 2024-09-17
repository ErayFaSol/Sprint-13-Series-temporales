from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import xgboost as xgb

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train, param_grid):
    model = lgb.LGBMRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_xgboost(X_train, y_train, param_grid):
    model = xgb.XGBRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
