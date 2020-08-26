import numpy as np
import joblib
import os
import time
import math
from config import cfg
from utils.dataset import load_npz
from utils.traj_speed import cubic_speed, timestamp
from utils.graphic import plt_show
# from sklearn.svm import SVR

# npz_list = os.listdir('./npz_502/')
model_dir = './model/'
test_dir = 'test_data'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

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
            if verbose:
                print(type(pos_raw_data[i]), type(pos_pre_data[i]))
            # print('after load what is pre',pos_raw_data[i][1], pos_pre_data[i])
            model = joblib.load(model_dir + '/' + model_name + str(i) + ".pkl")
            speed_data = cubic_speed(pos_raw_data[i])
            test_data = np.array([[speed_data[0][idx], speed_data[1][idx]] for idx in range(speed_data[0].shape[0])])[B:B+T]
            # test_data = np.array((speed_data[0], speed_data[1]))[B:B+T]
            for j in range(pos_raw_data[i].shape[0] - T - B):
                if verbose and j == 0:
                    print("raw speed", speed_data)
                    print("a",test_data)
                acc = model.predict(test_data)[T - 1]
                if verbose and j == 0:
                    print("acc", acc)
                vel = test_data[T - 1][1] + acc / F
                pre_x = test_data[T - 1][0] + test_data[T - 1][1] / F \
                        + acc / F / F / 2 # vt + 1/2 at^2
                if verbose and j == 0:
                    print("pre_x", pre_x)
                test_data[0:T - 1] = test_data[1:T]
                test_data[T - 1] = [pre_x, vel]
                if verbose and j == 0:
                    print("b", test_data)
                    print("pos_pre", pos_pre_data[i].shape)
                pos_pre_data[i][T+j+B] = pre_x
        if verbose:
            print("cost time:", time.time()-t_start)
        if show:
            # show_data = [
            #     [pos_raw_data[0],pos_raw_data[1],pos_raw_data[2],
            #     [pos_pre_data[0],pos_pre_data[1],pos_pre_data[2]]
            # ]
            plt_show([pos_raw_data, pos_pre_data], num=2, color=['red', 'green'])
        if record_err:
            # pos_raw_data[0][1][0][-1], pos_pre_data[0][-1]
            err = err_cal(np.array(pos_raw_data)[:,-1],
                        np.array(pos_pre_data)[:,-1])
            print("The error is %.2f%%"%(err))


def err_cal(xTrue, xPre):
    err_rate = 0.0
    for i in range(3):
        err_rate = err_rate + abs(xPre[i] - xTrue[i]) / abs(xTrue[i])
    return err_rate / 3.0 * 100.0

if __name__ == '__main__':
    eval_position()

