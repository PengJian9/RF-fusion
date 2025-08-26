import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor



def RF(X_train, X_test, Y_train, Y_test,tre_num=100):
    # 创建随机森林回归模型
    model = RandomForestRegressor(n_estimators=tre_num, max_depth=7, random_state=5)
    # 训练模型
    model.fit(X_train, Y_train)
    # 预测
    Y_pred = model.predict(X_test)
    # 计算 R^2 值
    r2 = r2_score(Y_test, Y_pred)
    rmse=np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae=np.mean(np.abs(Y_test-Y_pred))
    return r2,rmse,mae

def RF_group(X_train, X_test, Y_train, Y_test,tre_num=100):
    # 创建随机森林回归模型
    model = RandomForestRegressor(n_estimators=tre_num, max_depth=7, random_state=5)
    # 训练模型
    model.fit(X_train, Y_train)
    # 预测
    Y_pred = model.predict(X_test)

    return Y_pred

def Liner(X_train, X_test, Y_train, Y_test):
    # 创建线性回归模型
    model = LinearRegression()
    # 训练模型
    model.fit(X_train, Y_train)
    # 预测
    Y_pred = model.predict(X_test)
    # 计算 R^2 值
    r2 = r2_score(Y_test, Y_pred)
    rmse=np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae=np.mean(np.abs(Y_test-Y_pred))
    return r2,rmse,mae

def Liner_group(X_train, X_test, Y_train, Y_test):
    # 创建线性回归模型
    model = LinearRegression()
    # 训练模型
    model.fit(X_train, Y_train)
    # 预测
    Y_pred = model.predict(X_test)

    return Y_pred



def svm(X_train, X_test, Y_train, Y_test):
    # 创建支持向量机回归模型
    model = SVR(kernel='rbf')
    # 训练模型
    model.fit(X_train, Y_train)
    # 预测
    Y_pred = model.predict(X_test)
    # 计算 R^2 值
    r2 = r2_score(Y_test, Y_pred)
    rmse=np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae=np.mean(np.abs(Y_test-Y_pred))
    return r2,rmse,mae

def svm_group(X_train, X_test, Y_train, Y_test):
    # 创建支持向量机回归模型
    model = SVR(kernel='rbf')
    # 训练模型
    model.fit(X_train, Y_train)
    # 预测
    Y_pred = model.predict(X_test)

    return Y_pred

def poly(X_train, X_test, Y_train, Y_test):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    # 创建线性回归模型
    model = LinearRegression()
    model.fit(X_poly_train, Y_train)

    # 预测
    y_pred = model.predict(X_poly_test)

    # 评估模型
    rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
    mae = np.mean(np.abs(Y_test - y_pred))
    r2 = r2_score(Y_test, y_pred)
    return r2,rmse,mae

def poly_group(X_train, X_test, Y_train, Y_test):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    # 创建线性回归模型
    model = LinearRegression()
    model.fit(X_poly_train, Y_train)

    # 预测
    y_pred = model.predict(X_poly_test)

    return y_pred

def DT(X_train, X_test, Y_train, Y_test,depth=10):
    # 创建决策树回归模型
    model = DecisionTreeRegressor(max_depth=depth)
    # 训练模型
    model.fit(X_train, Y_train)
    # 预测
    Y_pred = model.predict(X_test)
    # 计算 R^2 值
    r2 = r2_score(Y_test, Y_pred)
    rmse=np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae=np.mean(np.abs(Y_test-Y_pred))
    return r2,rmse,mae

def DT_group(X_train, X_test, Y_train, Y_test,depth=10):
    # 创建决策树回归模型
    model = DecisionTreeRegressor(max_depth=depth)
    # 训练模型
    model.fit(X_train, Y_train)
    # 预测
    Y_pred = model.predict(X_test)

    return Y_pred

def xgboost(X_train, X_test, Y_train, Y_test):
    model = XGBRegressor(n_estimators=30, 
                                    max_depth=4, 
                                    eta=0.1, 
                                    subsample=0.7, 
                                    colsample_bytree=0.8,)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    r2 = r2_score(Y_test, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae = np.mean(np.abs(Y_test - Y_pred))
    return r2, rmse, mae

def xgboost_group(X_train, X_test, Y_train, Y_test):
    model = XGBRegressor(n_estimators=30, 
                                    max_depth=4, 
                                    eta=0.1, 
                                    subsample=0.7, 
                                    colsample_bytree=0.8,)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    return Y_pred