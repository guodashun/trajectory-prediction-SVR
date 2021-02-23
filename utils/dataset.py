import numpy as np
import pandas as pd
import os
import sys
sys.path.append("..")
from config import cfg

# data = pd.read_csv("", sep=',')
# data_txt = ""


# deprecated
def load_data(data_dir, dms_idx):
    print("Warning: This interface is deprecated, please use \"load_npz\" instead!")
    trajs = os.listdir(data_dir)
    objs = []
    ts = []
    for traj in trajs:
        raw_data = pd.read_csv("traj/" + traj)
        obj = raw_data[raw_data.columns[dms_idx + 2]].to_numpy()
        t = np.arange(0,len(obj)*1/cfg['frame_rate'],1/cfg['frame_rate']).reshape(-1,1)
        # print("traj_shape", obj.shape, t.shape)
        # print(len(obj), len(t))
        # break
        objs.append(obj)
        ts.append(t)
    objs = np.array(objs, dtype=object)
    ts = np.array(ts, dtype=object)
    return [ts, objs]


def load_npz(dir, traj_name):
    raw_data = np.load(dir + "/" + traj_name)
    obj = raw_data['position']
    return obj


def csv2npz(data, data_txt):
    chip = np.loadtxt(data_txt)
    # print(chip.shape)
    for i in range(chip.shape[0]):
        new_data = data[int(chip[i][0]):int(chip[i][1])]
        np.savez("" + str(chip[i][0]).zfill(5) + ".npz", frame_num=new_data["Frame"],time_step=new_data["Time"],position=new_data[["X", "Y", "Z"]],quaternion=new_data[["A", "B", "C", "D"]])
