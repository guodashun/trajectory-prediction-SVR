import numpy as np
import joblib
import os
from dataset import load_npz
from traj_speed import cubic_speed, cubic_speed_single
import time
import math
from utils.graphic import plt_show
# from sklearn.svm import SVR

# npz_list = os.listdir('./npz_502/')
model_dir = './model/'
model_list = os.listdir(model_dir)
frame_rate = 120
time_start = 0
time_step = 10
verbose = False


def eval_position():
    pos_raw_data = [0]*3
    pos_pre_data = [0]*3
    t_start = time.time()
    for i in range(3): # x,y,z
        pos_raw_data[i] = load_npz("test_data", i, frame_rate)
        pos_pre_data[i] = pos_raw_data[i][1][0].copy()
        # print('after load what is pre',pos_raw_data[i][1], pos_pre_data[i])
        model = joblib.load(model_dir + model_list[i])
        speed_data = cubic_speed(pos_raw_data[i])
        test_data = np.array(speed_data[0])[time_start:time_start+time_step]
        for j in range(pos_raw_data[i][0].shape[1] - time_step-time_start):
            if verbose and j == 1:
                print("raw speed", speed_data)
                print("a",test_data)
            acc = model.predict(test_data)[time_step - 1]
            if verbose and j == 1:
                print("acc", acc)
            vel = test_data[time_step - 1][1] + acc / frame_rate
            pre_x = test_data[time_step - 1][0] + test_data[time_step - 1][1] / frame_rate \
                    + acc / frame_rate / frame_rate / 2 # vt + 1/2 at^2
            if verbose and j == 1:
                print("pre_x", pre_x)
            test_data[0:time_step - 1] = test_data[1:time_step]
            test_data[time_step - 1] = [pre_x, vel]
            if verbose and j == 1:
                print("b", test_data)
            pos_pre_data[i][time_step+j+time_start] = pre_x
    # print('before plot', pos_raw_data)
    if verbose:
        print("cost time:", time.time()-t_start)
    show_data = [
        [pos_raw_data[0][1][0],pos_raw_data[1][1][0],pos_raw_data[2][1][0]],
        [pos_pre_data[0],pos_pre_data[1],pos_pre_data[2]]
    ]
    plt_show(show_data, num=2, color=['red', 'green'])


if __name__ == '__main__':
    eval_position()

