import numpy as np
import joblib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import os
from dataset import load_npz
from traj_speed import cubic_speed
# from sklearn.svm import SVR

# npz_list = os.listdir('./npz_502/')
model_dir = './model/'
model_list = os.listdir(model_dir)
frame_rate = 120
time_start = 0
time_step = 10
verbose = False


def eval_position():
    ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
    ax.set_title('3d_image_show')  # 设置本图名称
    ax.set_xlabel('X')  # 设置x坐标轴
    ax.set_ylabel('Y')  # 设置y坐标轴
    ax.set_zlabel('Z')  # 设置z坐标轴
    pos_raw_data = [0]*3
    pos_pre_data = [0]*3
    for i in range(3): # x,y,z
        # print('before load', pos_raw_data)s
        pos_raw_data[i] = load_npz("test_data", i, frame_rate)
        pos_pre_data[i] = pos_raw_data[i][1][0].copy()
        # print('after load what is pre',pos_raw_data[i][1], pos_pre_data[i])
        model = joblib.load(model_dir + model_list[i])
        # print('fuck', pos_raw_data[i])
        # print('fuck',np.array(pos_raw_data[i]).shape)
        speed_data = cubic_speed(pos_raw_data[i])
        # print('fuck speed shape', np.array(speed_data).shape, np.array(speed_data[0]).shape)
        # print('fuck speed', speed_data)
        test_data = np.array(speed_data[0])[time_start:time_start+time_step] # need test
        # print('fuck test shape',test_data.shape)
        # print(pos_raw_data[i][0].shape)
        for j in range(pos_raw_data[i][0].shape[1] - time_step-time_start):
            if verbose and j == 1:
                print("raw speed", speed_data)
                print("a",test_data)
            acc = model.predict(test_data)[time_step - 1]
            if verbose and j == 1:
                print("acc", acc)
            vel = test_data[time_step - 1][1] + acc / frame_rate
            pre_x = test_data[time_step - 1][0] + test_data[time_step - 1][1] / frame_rate \
                    + acc / frame_rate / frame_rate / 2 # vt * 1/2 at2
            if verbose and j == 1:
                print("pre_x", pre_x)
            test_data[0:time_step - 1] = test_data[1:time_step]
            test_data[time_step - 1] = [pre_x, vel]
            if verbose and j == 1:
                print("b", test_data)
            pos_pre_data[i][time_step+j+time_start] = pre_x
    # print('before plot', pos_raw_data)
    ax.plot(pos_raw_data[0][1][0],pos_raw_data[1][1][0],pos_raw_data[2][1][0], color='red')
    ax.plot(pos_pre_data[0],pos_pre_data[1],pos_pre_data[2], color='green')
    plt.show()


if __name__ == '__main__':
    eval_position()