import numpy as np
import joblib
import os
import time
import math
from config import cfg
from utils.dataset import load_npz
from utils.utils import cubic_speed, intergral_x, err_norm, plt_show
from tqdm import tqdm
# from sklearn.svm import SVR

# npz_list = os.listdir('./npz_502/')
model_dir = 'model'
object_name = "Green"
test_dir = f"npz_aug_scp/{object_name}/test_q"
test_list = os.listdir(test_dir)
if '.keep' in test_list:
    test_list.remove(".keep")

B = cfg['init_start']
F = cfg['frame_rate']
verbose = False
show = False

def eval_position(traj, T=None):
    if T == None:
        T = cfg['init_time']
    t_start = time.time()

    pos_raw_data = load_npz(test_dir,traj)
    pos_pre_data = pos_raw_data.copy()
    speed_raw_data = cubic_speed(pos_raw_data[:,B:B+T])
    speed_pre_data = speed_raw_data.copy()
    # print("speed_raw_data", speed_raw_data)
    # test_data = np.array([np.array([speed_raw_data[0,:,idx], speed_raw_data[1,:,idx]]).flatten() for idx in range(speed_raw_data[0][0].shape[0])])[B:B+T]
    # print("test_data", test_data.shape)
    test_data = np.array([speed_raw_data[0,:,T-1], speed_raw_data[1,:,T-1]]).flatten()
    traj_long = pos_raw_data.shape[0]

    for j in range(pos_raw_data.shape[0] - T - B):
        pre_x = [0]*3
        vel = [0]*3
        acc = [0]*3
        for i in range(3): # x,y,z    
            model = joblib.load(f"{model_dir}/{object_name}_{i}.pkl")
            acc[i] = model.predict(test_data.reshape(1,-1))
            pre_x[i], vel[i] = intergral_x(test_data[i], test_data[i+3], acc[i], 1/F)
        # test_data[0:T - 1] = test_data[1:T]
        test_data = np.array([pre_x, vel]).flatten()
        pos_pre_data[B+T+j] = pre_x
        speed_pre_data[:,:,B+T+j] = np.array([pre_x, vel, acc]).reshape(3,3)

    if verbose:
        print("cost time:", time.time()-t_start)
    
    # pos_raw_data[0][1][0][-1], pos_pre_data[0][-1]
    # print("pos",np.array(pos_raw_data)[-1], np.array(pos_pre_data)[-1],np.array(pos_raw_data)[-1]-np.array(pos_pre_data)[-1], np.linalg.norm(np.array(pos_raw_data)[-1]-np.array(pos_pre_data)[-1]) )
    pos_err = err_norm(np.array(pos_raw_data)[-1], np.array(pos_pre_data)[-1])
    acc_err = err_norm(speed_raw_data[2,:,-1], speed_pre_data[2,:,-1])

    
    if show:
        plt_show([pos_raw_data, pos_pre_data], num=2, color=['red', 'green'])
       
    return pos_err, acc_err, traj_long


if __name__ == '__main__':
    pos_errs = []
    acc_errs = []
    for traj in tqdm(test_list):
        pos_err, acc_err, _ = eval_position(traj)
        # print(f"The pos error is {pos_err}, acc error is {acc_err}")
        pos_errs.append(pos_err)
        acc_errs.append(acc_err)
    pos_errs = np.array(pos_errs)
    acc_errs = np.array(acc_errs)
    res = open("results.txt", 'a')
    res.write(f"{object_name}_acc_err:{acc_errs.max()},{acc_errs.min()},{acc_errs.mean()}\n")
    
    ach_pre_t = []

    for traj in tqdm(test_list):
        pos_err = 9999
        traj_long = 9999
        T = cfg['init_time']-1
        while(pos_err > 0.01 and T < traj_long-1):
            T += 1
            pos_err, _, traj_long = eval_position(traj, T)
        ach_pre_t.append((traj_long - T)/120.)
    ach_pre_t = np.array(ach_pre_t)

    res.write(f"{object_name}_t:{ach_pre_t.max()},{ach_pre_t.min()},{ach_pre_t.mean()}\n")
    res.close()
        