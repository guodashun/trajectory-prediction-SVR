import numpy as np
import pandas as pd
import os

# data = pd.read_csv("", sep=',')
# data_txt = ""


def load_data(data_dir, dms_idx, frame_rate):
    trajs = os.listdir(data_dir)
    objs = []
    ts = []
    for traj in trajs:
        raw_data = pd.read_csv("traj/" + traj)
        obj = raw_data[raw_data.columns[dms_idx + 2]].to_numpy()
        t = np.arange(0,len(obj)*1/frame_rate,1/frame_rate).reshape(-1,1)
        # print("traj_shape", obj.shape, t.shape)
        # print(len(obj), len(t))
        # break
        objs.append(obj)
        ts.append(t)
    objs = np.array(objs, dtype=object)
    ts = np.array(ts, dtype=object)
    return [ts, objs]


def load_npz(data_dir, idx, frame_rate):
    trajs = os.listdir(data_dir)
    objs = []
    ts = []
    for traj in trajs:
        raw_data = np.load(data_dir + "/" + traj)
        obj = raw_data['position'][:, idx]
        t = []
        # t = np.arange(0,len(obj)*(1/frame_rate),1/frame_rate).reshape(-1,1)
        for i in range(obj.shape[0]):
            t.append(i/frame_rate)
        # t = np.array(t, dtype=object).reshape(-1,1)
        t = np.array(t, dtype=object)
        # print("traj_shape", obj.shape, t.shape)
        # print(len(obj), len(t))
        # break
        objs.append(obj)
        ts.append(t)
    objs = np.array(objs, dtype=object)
    ts = np.array(ts, dtype=object)
    # print(objs.shape, ts.shape)
    return [ts, objs]


def csv2npz(data, data_txt):
    chip = np.loadtxt(data_txt)
    # print(chip.shape)
    for i in range(chip.shape[0]):
        new_data = data[int(chip[i][0]):int(chip[i][1])]
        np.savez("" + str(last).zfill(5) + ".npz", frame_num=new_data["Frame"],time_step=new_data["Time"],position=new_data[["X", "Y", "Z"]],quaternion=new_data[["A", "B", "C", "D"]])
