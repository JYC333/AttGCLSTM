import gc
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import ConvLSTM2D, Dense

from customize_layers.attconvlstm_layer import AttConvLSTM2D
from customize_layers.attgclstm_layer import AttGCLSTM2D
from customize_layers.gclstm_layer import GCLSTM2D
from utils import *

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def model_conv(time_step, n_station, n_feature, n_var, channels):
    input = keras.Input(shape=(time_step, n_station, n_feature + n_var, channels))
    h = ConvLSTM2D(1, 3, padding="same", return_sequences=True)(input)
    output = ConvLSTM2D(1, 3, padding="same", return_sequences=False)(h)

    log_dir = "E:\\logs\\ConvLSTM" + str(int(2 * 60 / time_step))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_images=True
    )
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=10)

    model = keras.Model(inputs=input, outputs=output)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mse", "mae"],
    )

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


def model_attconv(time_step, n_station, n_feature, n_var, channels):
    input = keras.Input(shape=(time_step, n_station, n_feature + n_var, channels))
    h = AttConvLSTM2D(1, 3, padding="same", return_sequences=True)(input)
    output = AttConvLSTM2D(1, 3, padding="same", return_sequences=False)(h)

    log_dir = "E:\\logs\\AttConvLSTM" + str(int(2 * 60 / time_step))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_images=True
    )
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=10)

    model = keras.Model(inputs=input, outputs=output)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mse", "mae"],
    )

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


def model_gc(time_step, n_station, n_feature, n_var, channels):
    input = keras.Input(shape=(time_step, n_station, n_feature + n_var, channels))
    h = GCLSTM2D(n_station, return_sequences=True)(input)
    output = GCLSTM2D(n_feature, return_sequences=False)(h)

    log_dir = "E:\\logs\\GCLSTM" + str(int(2 * 60 / time_step))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_images=True
    )
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=10)

    model = keras.Model(inputs=input, outputs=output)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mse", "mae"],
    )

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


def model_attgc(time_step, n_station, n_feature, n_var, channels):
    input = keras.Input(shape=(time_step, n_station, n_feature + n_var, channels))
    h = AttGCLSTM2D(HIDDEN_UNIT, return_sequences=True)(input)
    output = AttGCLSTM2D(n_feature, return_sequences=False)(h)

    log_dir = "E:\\logs\\AttGCLSTM" + str(int(2 * 60 / time_step))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_images=True
    )
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=10)

    model = keras.Model(inputs=input, outputs=output)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mse", "mae"],
    )

    model.fit(
        X_train,
        y_train,
        epochs=EPOCH,
        verbose=2,
        validation_split=0.25,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, tensorboard_callback],
    )
    preds = model.predict(X_test)

    del model, input, output, h
    gc.collect()

    return preds


def model_seqgc(time_step, n_station, n_feature, n_var, channels):
    input = keras.Input(shape=(time_step, n_station, n_feature + n_var, channels))
    encode_output, state_h, state_c = GCLSTM2D(
        n_feature, return_sequences=True, return_state=True
    )(input)
    output = GCLSTM2D(n_feature)(encode_output, initial_state=[state_h, state_c])

    log_dir = "E:\\logs\\SeqGCLSTM" + str(int(2 * 60 / time_step))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_images=True
    )
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=10)

    model = keras.Model(inputs=input, outputs=output)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mse", "mae"],
    )

    model.fit(
        X_train,
        y_train,
        epochs=EPOCH,
        validation_split=0.25,
        callbacks=[early_stopping, tensorboard_callback],
    )
    preds = model.predict(X_test)

    del model, input, output
    gc.collect()

    return preds


def model_seqattgc(time_step, n_station, n_feature, n_var, channels):
    input = keras.Input(shape=(time_step, n_station, n_feature + n_var, channels))
    h = AttGCLSTM2D(n_station, return_sequences=True)(input)
    encode_output, state_h, state_c = AttGCLSTM2D(
        n_station, return_sequences=True, return_state=True
    )(h)
    h = AttGCLSTM2D(n_station, return_sequences=True)(
        encode_output, initial_state=[state_h, state_c]
    )
    output = AttGCLSTM2D(n_feature)(h)

    log_dir = "E:\\logs\\SeqAttGCLSTM" + str(int(2 * 60 / time_step))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, write_images=True
    )
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.000001, patience=10)

    model = keras.Model(inputs=input, outputs=output)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mse", "mae"],
    )

    model.fit(
        X_train,
        y_train,
        epochs=EPOCH,
        validation_split=0.25,
        verbose=2,
        callbacks=[early_stopping, tensorboard_callback],
    )
    preds = model.predict(X_test)

    del model, input, output, h, encode_output, state_h, state_c
    gc.collect()

    return preds


Use_Var = 1
channels = 1
EPOCH = 200

if __name__ == "__main__":
    if Use_Var:
        # l1 = [0, 1, 2, 3]
        # vars_list = []
        # for i in range(1, len(l1) + 1):
        #     iter = itertools.combinations(l1, i)
        #     vars_list += list(iter)
        # vars_list = [(0,), (1,), (2,), (3,), (0, 1, 2, 3)]
        vars_list = [(0, 2, 3)]
    else:
        vars_list = [1]

    # 'ConvLSTM': model_conv, 'AttConvLSTM': model_attconv,'GCLSTM': model_gc,'AttGCLSTM': model_attgc,  'SeqAttGCLSTM': model_seqattgc,'SeqGCLSTM': model_seqgc
    model = {"AttGCLSTM": model_attgc}

    for key in model.keys():
        print(key)
        var_list = vars_list
        for var_use in var_list:
            times = [60]  # 10, 15, 20, 30, 60
            for t in times:
                print("time interval:", t)
                path = "Docked_" + str(t) + "/"
                time_inter = int(24 * 60 / t)
                train = time_inter * 16
                test = time_inter * 5

                raw_data = np.load(path + "Docked_Station_adj_weight" + str(t) + ".npy")
                n_station = raw_data.shape[1]
                raw_data_x = np.sum(raw_data, axis=1).reshape(-1, n_station, 1)
                raw_data_y = np.sum(raw_data, axis=2).reshape(-1, n_station, 1)
                raw_data = np.concatenate((raw_data_x, raw_data_y), axis=2)
                del raw_data_x, raw_data_y
                gc.collect()
                n_feature = raw_data.shape[-1]

                raw_data_sqrt = np.sqrt(raw_data)

                diff_normal = difference(raw_data_sqrt, 1)
                diff_season = difference(diff_normal, time_inter)
                diff_season = diff_season.reshape((-1, n_station * n_feature))

                scaler = MinMaxScaler(feature_range=(0, 1))
                diff_season = scaler.fit_transform(diff_season)

                diff_season = diff_season.reshape(-1, n_station, n_feature, 1)

                seq_train = diff_season[:train]
                seq_test = diff_season[train:]

                for t_s in [i for i in range(2, 7)]:
                    time_step = int(t_s * 60 / t)

                    X_train, y_train = seq_to_training_data(seq_train, time_step)
                    X_test, y_test = seq_to_training_data(seq_test, time_step)

                    if Use_Var and var_use != None:
                        if len(var_use) == 1:
                            Vars = Generate_Var(t, n_station, train + test)[var_use[0]]
                        else:
                            Vars = Generate_Var(t, n_station, train + test)
                            Vars_tem = Vars[var_use[0]]
                            for j in var_use[1:]:
                                Vars_tem = np.concatenate((Vars_tem, Vars[j]), axis=2)
                            Vars = Vars_tem
                            del Vars_tem
                            gc.collect()

                        dense1 = Dense(50)
                        dense2 = Dense(15)
                        dense3 = Dense(1)
                        Vars = dense3(Vars)
                        Vars = np.array(Vars)

                        var_shape = Vars.shape[2]

                        Vars = Vars.reshape((-1, n_station * var_shape))

                        scaler_var = MinMaxScaler(feature_range=(0, 1))
                        Vars = scaler_var.fit_transform(Vars)

                        Vars = Vars.reshape(-1, n_station, var_shape, 1)

                        var_train = Vars[:train]
                        var_test = Vars[train:]

                        X_var_train, _ = seq_to_training_data(var_train, time_step)
                        X_var_test, _ = seq_to_training_data(var_test, time_step)

                        X_train = np.concatenate((X_train, X_var_train), axis=3)
                        X_test = np.concatenate((X_test, X_var_test), axis=3)
                    else:
                        var_shape = 0

                    for lr in [i / 1000 for i in range(5, 11)]:
                        LEARNING_RATE = lr
                        for b_s in [32, 64, 128]:
                            if lr == 0.005 and b_s <= 128:
                                continue
                            BATCH_SIZE = b_s
                            for h_u in [32, 64, 128, 256]:
                                if lr == 0.005 and b_s <= 128 and h_u <= 64:  #
                                    continue
                                HIDDEN_UNIT = h_u
                                RMSE = []
                                MAE = []
                                Running_Time = []
                                for j in range(100):
                                    t0 = time.process_time()
                                    preds = model[key](
                                        time_step,
                                        n_station,
                                        n_feature,
                                        var_shape,
                                        channels,
                                    )
                                    total_running_time = time.process_time() - t0

                                    preds = scaler.inverse_transform(
                                        preds.reshape(-1, n_station * n_feature)
                                    )
                                    preds = preds.reshape((-1, n_station, n_feature))

                                    preds = inverse_difference(
                                        diff_normal[train:], preds, time_inter
                                    )
                                    preds = inverse_difference(
                                        raw_data[train:], preds, 1
                                    )

                                    preds[preds < 0] = 0

                                    k = preds[0]
                                    preds[:-1] = preds[1:]
                                    preds[-1] = k

                                    preds = np.around(preds)

                                    np.save(path + key + f"Preds_merge_{t}.npy", preds)

                                    rmse = np.sqrt(
                                        np.average((raw_data[train:] - preds) ** 2)
                                    )
                                    mae = np.average(np.abs(raw_data[train:] - preds))
                                    RMSE.append(rmse)
                                    MAE.append(mae)
                                    Running_Time.append(total_running_time)

                                    del preds, k
                                    gc.collect()

                                if Use_Var and var_use != None:
                                    open(path + f"Result_{t}.csv", "a").write(
                                        f"{key} {(var_use).replace(',', '')},{np.min(RMSE)},{np.min(MAE)},{np.average(Running_Time)},{t_s},{lr},{b_s},{h_u}\n"
                                    )
                                else:
                                    open(path + "Result" + str(t) + ".csv", "a").write(
                                        f"{key},{np.average(RMSE)},{np.average(MAE)},{np.average(Running_Time)}\n"
                                    )

                    del X_train, y_train, X_test, y_test
                    if Use_Var and var_use != None:
                        del var_train, var_test
                        del X_var_train, X_var_test
                    gc.collect()

                del raw_data, raw_data_sqrt
                del diff_normal, diff_season
                del seq_train, seq_test, scaler
                gc.collect()
