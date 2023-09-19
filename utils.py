import numpy as np
import pandas as pd


def seq_to_training_data(seq, step):
    X = []
    y = []
    for i in range(len(seq) - step):
        X.append(seq[i : i + step])
        y.append(seq[i + step])
    for i in range(len(seq) - step + 1, len(seq) + 1):
        X.append(np.array(list(seq[i - 1 :]) + list(seq[: i + step - len(seq) - 1])))
        y.append(seq[i + step - len(seq)])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def difference(dataset, interval=1):
    diff = list()
    for i in range(len(dataset)):
        diff.append(dataset[i] - dataset[i - interval])
    return np.array(diff)


def inverse_difference(history, yhat, interval=1):
    inverse = list()
    for i in range(len(yhat)):
        inverse.append(yhat[i] + history[i - interval])
    return np.array(inverse)


def Generate_Var(time_i, nodes, total_sample):
    station = list(
        np.array(pd.read_csv("Docked_Station_Center.csv", header=None)).flatten()
    )

    with open("Docked_Station_Var_POI+Road.csv") as f:
        line = f.readline()
        line = f.readline()
        nums = len(line.split(",")) - 1
        POI = np.zeros((nodes, nums))
        while line:
            a = line.split(",")
            if int(a[0]) not in station:
                line = f.readline()
                continue
            k = station.index(int(a[0]))
            for i in range(nums):
                if i != nums - 1:
                    POI[k][i] = int(a[i + 1])
                else:
                    POI[k][i] = float(a[i + 1])
            line = f.readline()
    POI = np.tile(POI, (total_sample, 1, 1))

    with open("Docked_Station_Var_Personal.csv") as f:
        line = f.readline()
        line = f.readline()
        nums = len(line.split(",")) - 3
        Personal_All = np.zeros((nodes, nums))
        while line:
            a = line.split(",")
            if int(a[0]) not in station:
                line = f.readline()
                continue
            k = station.index(int(a[0]))
            for i in range(nums):
                Personal_All[k][i] = int(a[i + 3])
            line = f.readline()
    Personal_All = np.tile(Personal_All, (total_sample, 1, 1))

    path = "Docked_" + str(time_i) + "/"
    with open(path + "Weather" + str(time_i) + ".csv") as f:
        line = f.readline()
        line = f.readline()
        nums = len(line.split(","))
        Weather = np.zeros((total_sample, 1, nums))
        k = 0
        while line:
            a = line.split(",")
            for i in range(nums):
                if i in [1, 8, 9]:
                    Weather[k][0][i] = float(a[i])
                else:
                    Weather[k][0][i] = int(float(a[i]))
            k += 1
            line = f.readline()
    Weather = np.tile(Weather, (1, nodes, 1))

    Personal_Time = np.load(
        path + "Docked_Station_Personal_Time" + str(time_i) + ".npy"
    )
    Personal_Time = Personal_Time.astype("float32")

    return POI, Personal_All, Weather, Personal_Time
