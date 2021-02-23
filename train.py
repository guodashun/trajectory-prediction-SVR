from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from utils.dataset import load_data, load_npz
from utils.utils import cubic_speed
import joblib
import pandas as pd
import numpy as np
import os
from multiprocessing import cpu_count

# param_grid = {'C': np.linspace(0.1,3,30), 'gamma': np.linspace(0.01,0.6,60)}
param_grid = {'C': np.array([2**(i-5) for i in range(20)]), 'gamma': np.array([2**(i-15) for i in range(20)])}
# param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000], 'gamma': [10, 1, 0.1, 0.01, 0.001]}

dms_num = 3 # only pos

train_threads = cpu_count() - 4 # for stable train
save_path = 'model'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def train_pos(dms_idx, object_name):
    
    traj_dir = f"npz_aug_scp/{object_name}/train_q"
    traj_list = os.listdir(traj_dir)
    model = SVR(kernel='rbf')
    X = [] # zeta = {[x; v]}
    y = [] # Y = {a}
    for i,traj in enumerate(traj_list):
        # choose the raw data
        if (i+1) % 20 == 0:
            traj_data = load_npz(traj_dir, traj)
            xva = cubic_speed(traj_data)
            for i in range(xva.shape[2]):
                X.append(xva[0:2,:,i].reshape(2*dms_num))
                y.append(xva[2,dms_idx,i])
    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)
    
    grid_search = GridSearchCV(model, param_grid, n_jobs = train_threads, verbose=1)
    grid_search.fit(X, y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVR(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])
    model.fit(X, y)

    joblib.dump(model, f"{save_path}/{object_name}_{dms_idx}.pkl")
    param_f = open(f"{save_path}/params.txt", "a")
    param_f.write(f"{object_name}_{dms_idx}: C:{best_parameters['C']} gamma:{best_parameters['gamma']}\n")
    param_f.close()


if __name__ == '__main__':
    for j in ["Gourd","Banana"]:
        for i in range(dms_num):
            train_pos(i, j)
