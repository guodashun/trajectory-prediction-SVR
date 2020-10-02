from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from utils.dataset import load_data, load_npz
from utils.utils import cubic_speed
import joblib
import pandas as pd
import numpy as np
import os
from multiprocessing import cpu_count

param_grid = {'C': np.linspace(0.1,3,30), 'gamma': np.linspace(0.01,0.6,60)}
# param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000], 'gamma': [10, 1, 0.1, 0.01, 0.001]}

traj_dir = 'npz_aug2'
traj_list = os.listdir(traj_dir)
train_threads = cpu_count()
save_path = 'model'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def train_pos(dms_idx):
    model = SVR(kernel='rbf')
    # trajs_data = load_data("traj", dms_idx, frame_rate)
    X = [] # zeta = {[x, v]}
    y = [] # Y = {a}
    for traj in traj_list:
        traj_data = load_npz(traj_dir, traj, dms_idx)
        xva = cubic_speed(traj_data)
        for i in range(xva.shape[1]):
            X.append([xva[0, i], xva[1, i]])
            y.append(xva[2,i])
    
    grid_search = GridSearchCV(model, param_grid, n_jobs = train_threads, verbose=1)
    grid_search.fit(X, y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVR(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])
    model.fit(X, y)
    joblib.dump(model, save_path + '/dms_' + str(dms_idx) + '.pkl')


if __name__ == '__main__':
    for i in range(3):
        train_pos(i)
