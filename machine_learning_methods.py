import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from utils import *


def model_svm(X_train, y_train, X_test):
    clf = SVR(kernel="rbf")
    clf.fit(X_train, y_train.ravel())

    preds = clf.predict(X_test)
    return preds


def model_xgboost(X_train, y_train, X_test):
    xg_reg = xgb.XGBRegressor(objective="reg:squarederror")
    xg_reg.fit(X_train, y_train)

    preds = xg_reg.predict(X_test)
    return preds


if __name__ == "main":
    model = {"SVM": model_svm, "XGBOOST": model_xgboost}

    for key in model.keys():
        times = [120]  # 10, 15, 20, 30, 60
        for t in times:
            print("time interval:", t)
            path = "Docked_" + str(t) + "/"
            time_inter = int(24 * 60 / t)
            train = time_inter * 14
            test = time_inter * 7
            time_step = int(2 * 60 / t)

            data = np.load(path + "Docked_Station_adj_weight" + str(t) + ".npy")
            shape = data.shape[1]
            raw_data_x = np.sum(data, axis=1).reshape(-1, shape, 1)
            raw_data_y = np.sum(data, axis=2).reshape(-1, shape, 1)
            data = np.concatenate((raw_data_x, raw_data_y), axis=2)
            data = data.reshape((data.shape[0], -1))
            RMSE = np.array([])
            MAE = np.array([])
            t0 = time.process_time()
            for i in range(len(data[0])):
                raw_data = data[:, i]
                raw_data_sqrt = np.sqrt(raw_data)

                diff_normal = difference(raw_data_sqrt, 1)
                diff_season = difference(diff_normal, time_inter)

                seq_train = diff_season[:train]
                seq_test = diff_season[train:]

                scaler = MinMaxScaler(feature_range=(0, 1))
                seq_train = scaler.fit_transform(seq_train.reshape((-1, 1)))
                seq_test = scaler.transform(seq_test.reshape((-1, 1)))

                X_train, y_train = seq_to_training_data(seq_train, time_step)
                X_test, y_test = seq_to_training_data(seq_test, time_step)
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

                preds = model[key](X_train, y_train, X_test)

                preds = scaler.inverse_transform(preds.reshape((-1, 1)))
                preds = inverse_difference(diff_normal[train:], preds, time_inter)
                preds = inverse_difference(raw_data[train:], preds, 1)

                preds[preds < 0] = 0

                k = preds[0]
                preds[:-1] = preds[1:]
                preds[-1] = k

                preds = np.around(preds)

                mse = mean_squared_error(raw_data[train:], preds)
                mae = mean_absolute_error(raw_data[train:], preds)
                RMSE = np.append(RMSE, mse)
                MAE = np.append(MAE, mae)

            RMSE = np.sqrt(np.average(RMSE))
            MAE = np.average(MAE)
            open(path + f"Result_{t}.csv", "a").write(f"{key},{RMSE},{MAE}\n")
            print(time.process_time() - t0)
