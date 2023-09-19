import gc
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Flatten

from utils import *

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


LEARNING_RATE = 0.0001
EPOCH = 200


def model_ann(input_shape, hidden_shape, X_train, y_train, X_test):
    input = keras.Input(shape=input_shape)
    h = Flatten()(input)
    h = Dense(hidden_shape)(h)
    output = Dense(input_shape[1])(h)

    model = keras.Model(inputs=input, outputs=output)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mse", "mae"],
    )

    log_dir = "logs\\ANN" + str(t)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_images=True,
    )
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=10)

    model.fit(
        X_train,
        y_train,
        epochs=EPOCH,
        validation_split=0.25,
        callbacks=[early_stopping, tensorboard_callback],
    )
    preds = model.predict(X_test)

    del model, input, output, h
    gc.collect()

    return preds


def model_lstm(input_shape, hidden_shape, X_train, y_train, X_test):
    input = keras.Input(shape=input_shape)
    h = LSTM(hidden_shape, return_sequences=True)(input)
    output = LSTM(input_shape[1])(h)

    model = keras.Model(inputs=input, outputs=output)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mse", "mae"],
    )

    log_dir = "E:\\logs\\LSTM" + str(t)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_images=True
    )
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=30)

    model.fit(
        X_train,
        y_train,
        epochs=EPOCH,
        validation_split=0.25,
        verbose=2,
        callbacks=[early_stopping, tensorboard_callback],
    )
    preds = model.predict(X_test)

    del model, input, output, h
    gc.collect()

    return preds


if __name__ == "main":
    model = {"ANN": model_ann, "LSTM": model_lstm}

    for key in model.keys():
        times = [60]  # 10, 15, 20, 30, 60
        for t in times:
            print("time interval:", t)
            path = "Docked_" + str(t) + "/"
            time_inter = int(24 * 60 / t)
            train = time_inter * 16
            test = time_inter * 5
            time_step = int(2 * 60 / t)

            raw_data = np.load(path + "Docked_Station_adj_weight" + str(t) + ".npy")
            shape = raw_data.shape[1]
            raw_data_x = np.sum(raw_data, axis=1).reshape(-1, shape, 1)
            raw_data_y = np.sum(raw_data, axis=2).reshape(-1, shape, 1)
            raw_data = np.concatenate((raw_data_x, raw_data_y), axis=2)
            raw_data = raw_data.reshape((raw_data.shape[0], -1))
            del raw_data_x, raw_data_y
            gc.collect()

            raw_data_sqrt = np.sqrt(raw_data)

            diff_normal = difference(raw_data_sqrt, 1)
            diff_season = difference(diff_normal, time_inter)

            seq_train = diff_season[:train]
            seq_test = diff_season[train:]

            scaler = MinMaxScaler(feature_range=(0, 1))
            seq_train = scaler.fit_transform(seq_train)
            seq_test = scaler.transform(seq_test)

            X_train, y_train = seq_to_training_data(seq_train, time_step)
            X_test, y_test = seq_to_training_data(seq_test, time_step)

            RMSE = []
            MAE = []
            Running_Time = []
            for j in range(10):
                t0 = time.process_time()
                preds = model[key](
                    (time_step, X_train.shape[-1]), shape, X_train, y_train, X_test
                )
                total_running_time = time.process_time() - t0

                preds = scaler.inverse_transform(preds)

                preds = inverse_difference(diff_normal[train:], preds, time_inter)
                preds = inverse_difference(raw_data[train:], preds, 1)

                preds[preds < 0] = 0

                k = preds[0]
                preds[:-1] = preds[1:]
                preds[-1] = k

                preds = np.around(preds)

                rmse = np.sqrt(mean_squared_error(raw_data[train:], preds))
                mae = mean_absolute_error(raw_data[train:], preds)
                RMSE.append(rmse)
                MAE.append(mae)
                Running_Time.append(total_running_time)

            del raw_data, raw_data_sqrt
            del diff_normal, diff_season
            del seq_train, seq_test, scaler
            del X_train, y_train, X_test, y_test
            del preds, k
            gc.collect()

            open(path + f"Result_{t}.csv", "a").write(
                f"{key},{np.average(RMSE)},{np.average(MAE)},{np.average(Running_Time)}\n"
            )
