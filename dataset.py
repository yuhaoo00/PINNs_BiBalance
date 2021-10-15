import matplotlib.pyplot as plt
import torch
import numpy as np
from problem import *

class Trainset(object):
    def __init__(self, problem, *args):
        self.problem = problem
        self.domain = problem.domain
        self.args = args

    def __call__(self, device, plot=False, verbose='tensor'):
        n_t, n_x = self.args[0], self.args[1]
        x1_r, x1_t, x1_x0, x1_x1 = self._uniform_sample(n_t, n_x[0], 1)
        x2_r, x2_t, x2_x0, x2_x1 = self._uniform_sample(n_t, n_x[1], 2)
        x3_r, x3_t, x3_x0, x3_x1 = self._uniform_sample(n_t, n_x[2], 3)
        
        x_t = np.concatenate((x1_t,x2_t,x3_t),0)
        X = [x1_r, x1_t, x1_x0, x1_x1, x2_r, x2_t, x2_x0, x2_x1, x3_r, x3_t, x3_x0, x3_x1]
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(x_t[:, 0], x_t[:, 1], facecolor='b', s=1)
            ax.scatter(x1_r[:, 0], x1_r[:, 1], facecolor='r', s=1)
            ax.scatter(x2_r[:, 0], x2_r[:, 1], facecolor='r', s=1)
            ax.scatter(x3_r[:, 0], x3_r[:, 1], facecolor='r', s=1)
            ax.scatter(x1_x0[:, 0], x1_x0[:, 1], facecolor='g', s=1)
            ax.scatter(x2_x0[:, 0], x2_x0[:, 1], facecolor='g', s=1)
            ax.scatter(x3_x0[:, 0], x3_x0[:, 1], facecolor='g', s=1)
            ax.scatter(x1_x1[:, 0], x1_x1[:, 1], facecolor='y', s=1)
            ax.scatter(x2_x1[:, 0], x2_x1[:, 1], facecolor='y', s=1)
            ax.scatter(x3_x1[:, 0], x3_x1[:, 1], facecolor='y', s=1)
            
            #ax.set_aspect('equal')
            plt.show()

        if verbose == 'tensor':
            X_tensor = []
            for i in range(len(X)):
                X_tensor.append(torch.from_numpy(X[i]).float().to(device))
            X = X_tensor
            
        return X

    def _uniform_sample(self, n_t, n_x, part):
        tdomain = self.domain[0]
        xdomain = self.domain[part]
        t = np.linspace(tdomain[0], tdomain[1], n_t)
        x = np.linspace(xdomain[0], xdomain[1], n_x)
        t, x = np.meshgrid(t, x)
        tx = np.hstack((t.reshape(t.size, -1), x.reshape(x.size, -1)))

        mask_t = (tx[:, 0] - tdomain[0]) == 0
        mask_x0 = (tx[:, 1] - xdomain[0]) == 0
        mask_x1 = (tx[:, 1] - xdomain[1]) == 0
        x_t = tx[mask_t] #t=0
        x_x0 = tx[mask_x0] #x=min
        x_x1 = tx[mask_x1] #x=max

        mask_t[mask_x0 == True] = True
        mask_t[mask_x1 == True] = True
        mask = mask_t
        x_r = tx[np.logical_not(mask)] #res

        return x_r, x_t, x_x0, x_x1

class Testset(object):
    def __init__(self, problem, *args):
        self.problem = problem
        self.domain = problem.domain
        self.args = args

    def __repr__(self):
        return f'{self.__doc__}'

    def __call__(self, device, plot=False, verbose='tensor'):
        n_t, n_x = self.args[0], self.args[1]
        X1, t, x1 = self._uniform_sample(n_t, n_x[0], 1)
        X2, _, x2 = self._uniform_sample(n_t, n_x[1], 2)
        X3, _, x3 = self._uniform_sample(n_t, n_x[2], 3)
        x = np.concatenate((x1,x2,x3),0)
        t, x = np.meshgrid(t, x)
    
        if plot == True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(X1[:, 0], X1[:, 1], facecolor='r', s=1)
            ax.scatter(X2[:, 0], X2[:, 1], facecolor='r', s=1)
            ax.scatter(X3[:, 0], X3[:, 1], facecolor='r', s=1)
            plt.show()

        if verbose == 'tensor':
            X1 = torch.from_numpy(X1).float().to(device)
            X2 = torch.from_numpy(X2).float().to(device)
            X3 = torch.from_numpy(X3).float().to(device)

        return [X1,X2,X3], t, x
    
    def _uniform_sample(self, n_t, n_x, part):
        tdomain = self.domain[0]
        xdomain = self.domain[part]
        t = np.linspace(tdomain[0], tdomain[1], n_t)
        x = np.linspace(xdomain[0], xdomain[1], n_x)
        tm, xm = np.meshgrid(t, x)
        tx = np.hstack((tm.reshape(tm.size, -1), xm.reshape(xm.size, -1)))

        return tx, t, x

if __name__ == "__main__":
    problem = Problem([300*1377,862*2100,74.2*1726], [0.082,0.37,0.045],\
                    310.15*1e-3, 2, [40,8.496], 0.02, 0.01,\
                    np.array([[0,10.],[0,0.6],[0.7,1.45],[1.55,5.05]]))

    trainset = Trainset(problem, 30, [5,7,20])
    X = trainset(torch.device("cuda:0"), plot=True)
    
    testset = Testset(problem, 30, [5,7,20])
    X,t,x = testset(torch.device("cuda:0"), plot=True)
    print(t.shape)
    print(x.shape)