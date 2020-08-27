import numpy as np
import joblib
import os
import time
import math
from config import cfg
from utils.dataset import load_npz
from utils.utils import cubic_speed, intergral_x, err_cal, plt_show
# from sklearn.svm import SVR

# npz_list = os.listdir('./npz_502/')
model_dir = './model/'
test_dir = 'test_data'
model_name = "dms_"
test_list = os.listdir(test_dir)
test_list.remove(".keep")
T = cfg['init_time']
B = cfg['init_start']
F = cfg['frame_rate']
verbose = False
show = True
record_err = True


def eval_position():
    pos_raw_data = [0]*3
    pos_pre_data = [0]*3

    t_start = time.time()

    for traj in test_list:
        for i in range(3): # x,y,z
            pos_raw_data[i] = load_npz("test_data",traj, i)
            pos_pre_data[i] = pos_raw_data[i].copy()
            speed_data = cubic_speed(pos_raw_data[i])
            test_data = np.array([[speed_data[0][idx], speed_data[1][idx]] for idx in range(speed_data[0].shape[0])])[B:B+T]
            model = joblib.load(model_dir + '/' + model_name + str(i) + ".pkl")

            for j in range(pos_raw_data[i].shape[0] - T - B):
                acc = model.predict(test_data)[T - 1]
                pre_x, vel = intergral_x(test_data[T - 1][0], test_data[T - 1][1], acc, 1/F)
                test_data[0:T - 1] = test_data[1:T]
                test_data[T - 1] = [pre_x, vel]
                pos_pre_data[i][T+j+B] = pre_x
        
        if verbose:
            print("cost time:", time.time()-t_start)
        
        if show:
            plt_show([pos_raw_data, pos_pre_data], num=2, color=['red', 'green'])
       
        if record_err:
            # pos_raw_data[0][1][0][-1], pos_pre_data[0][-1]
            err = err_cal(np.array(pos_raw_data)[:,-1],
                        np.array(pos_pre_data)[:,-1])
            print("The error is %.2f%%"%(err))


if __name__ == '__main__':
    eval_position()

