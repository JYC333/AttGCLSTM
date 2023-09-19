import time

import numpy as np
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from utils import *

if __name__ == "main":
    times = [10, 15, 20, 30, 60]
    for t in times:
        path = "Docked_" + str(t) + "/"
        time_inter = int(24 * 60 / t)
        train = time_inter * 14
        test = time_inter * 7

        data = np.load(path + "Docked_Station_links" + str(t) + ".npy")
        RMSE = np.array([])
        t0 = time.process_time()
        for i in range(len(data[0])):
            raw_data = data[:, i]
            raw_data_sqrt = np.sqrt(raw_data)

            diff_normal = difference(raw_data_sqrt, 1)
            diff_season = difference(diff_normal, time_inter)

            scaler = MinMaxScaler(feature_range=(0, 1))
            diff_season = scaler.fit_transform(diff_season.reshape((-1, 1)))

            model = auto_arima(
                diff_season,
                start_p=1,
                start_q=1,
                max_p=9,
                max_q=6,
                max_d=3,
                max_order=None,
                seasonal=True,
                m=1,
                test="adf",
                trace=False,
                error_action="ignore",  # don't want to know if an order does not work
                suppress_warnings=True,  # don't want convergence warnings
                stepwise=True,
                information_criterion="bic",
                njob=-1,
            )
            fit_model = model.fit(diff_season)

            preds = fit_model.predict(start=train, end=train + test - 1, dynamic=True)
            preds = scaler.inverse_transform(preds.reshape((-1, 1)))
            preds = inverse_difference(diff_normal[train:], preds, time_inter)
            preds = inverse_difference(raw_data_sqrt[train:], preds, 1)
            mse = mean_squared_error(preds, raw_data[train:])
            RMSE = np.append(RMSE, mse)

        RMSE = np.sqrt(np.average(RMSE))
        open(path + "Result" + str(t) + ".csv", "a").write("SARIMA," + str(RMSE) + "\n")
        print(time.process_time() - t0)
