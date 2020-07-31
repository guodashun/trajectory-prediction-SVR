from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from dataset import load_data, load_npz
from traj_speed import cubic_speed
import joblib
import pandas as pd
import numpy as np
import os

frame_rate = 120
param_grid = {'C': np.linspace(0.1,3,30), 'gamma': np.linspace(0.01,0.6,60)}
# param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000], 'gamma': [10, 1, 0.1, 0.01, 0.001]}


def train_pos(dms_idx):
    model = SVR(kernel='rbf')
    # trajs_data = load_data("traj", dms_idx, frame_rate)
    trajs_data = load_npz('npz_502', dms_idx, frame_rate)
    train_data = cubic_speed(trajs_data)
    grid_search = GridSearchCV(model, param_grid, n_jobs = 16, verbose=1)

    # print(np.shape(trajs_data))
    # print(np.shape(trajs_data[0]), np.shape(trajs_data[1]))
    # print(trajs_data[0].shape, trajs_data[1].shape)
    # print(trajs_data[0], trajs_data[1])
    # print(trajs_data[0][0].ravel().shape, trajs_data[1][0].shape)
    # grid_search.fit(trajs_data[1][0], trajs_data[0][0])
    grid_search.fit(train_data[0], train_data[1])
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVR(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])
    model.fit(train_data[0], train_data[1])
    joblib.dump(model, 'model/dms_' + str(dms_idx) + '.pkl')


if __name__ == '__main__':
    for i in range(3):
        train_pos(i)
