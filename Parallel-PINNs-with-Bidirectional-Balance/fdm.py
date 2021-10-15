from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import os
plt.rc('font',family='Times New Roman')
matplotlib.use('Agg')

m1 = 50
m2 = 70
m3 = 200
m = m1+m2+m3
n = 200000

Tg = 2000
T0 = 310.15
conv = 40
h2 =  8.496
Ca = [300*1377,862*2100,74.2*1726]
lam = [0.082,0.37,0.045]
l = [0.6e-3,0.85e-3,3.6e-3]
t = 60

delta_x1 = l[0]/m1
delta_x2 = l[1]/m2
delta_x3 = l[2]/m3
dx = [delta_x1, delta_x2, delta_x3]
dt = t/n

r1 = dt/(delta_x1**2)*(lam[0]/Ca[0])
r2 = dt/(delta_x2**2)*(lam[1]/Ca[1])
r3 = dt/(delta_x3**2)*(lam[2]/Ca[2])

print(r1,r2,r3)


T = np.zeros((n,m)) + T0

for i in tqdm(range(n-1)):
    tmp = (conv*(Tg-T[i,0]) - lam[0]*(T[i,0]-T[i,1])/dx[0]) * dt/(0.5*dx[0]*Ca[0]) + T[i,0]
    T[i+1,0] = tmp

    for j in range(1, m1-1):
        tmp = lam[0]*(T[i,j+1]-2*T[i,j]+T[i,j-1])/dx[0]*dt/(dx[0]*Ca[0])+T[i,j]
        T[i+1,j] = tmp
    tmp = (lam[1]*(T[i,m1]-T[i,m1-1])/dx[1] + lam[0]*(T[i,m1-2]-T[i,m1-1])/dx[0])*\
            dt/(0.5*(dx[0]*Ca[0]+dx[1]*Ca[1])) + T[i,m1-1]
    T[i+1,m1-1] = tmp

    for j in range(m1, m1+m2-1):
        tmp = lam[1]*(T[i,j+1]-2*T[i,j]+T[i,j-1])/dx[1]*dt/(dx[1]*Ca[1])+T[i,j] 
        T[i+1,j]=tmp 
    tmp = (lam[2]*(T[i,m1+m2]-T[i,m1+m2-1])/dx[2] + lam[1]*(T[i,m1+m2-2]-T[i,m1+m2-1])/dx[1])*\
            dt/(0.5*(dx[1]*Ca[1]+dx[2]*Ca[2])) + T[i,m1+m2-1]
    T[i+1,m1+m2-1] = tmp

    for j in range(m1+m2, m1+m2+m3-1):
        tmp = lam[2]*(T[i,j+1]-2*T[i,j]+T[i,j-1])/dx[2]*dt/(dx[2]*Ca[2])+T[i,j]
        T[i+1,j] = tmp 
    tmp = (lam[2]*(T[i,m-2]-T[i,m-1])/dx[2] - h2*(T[i,m-1]-T0))*\
            dt/(0.5*(dx[2]*Ca[2])) + T[i,m-1]
    T[i+1,m-1] = tmp

T = T.T
cols = np.linspace(0, n-1, 300).astype(int)
T = T[:,cols]
print(T.shape)
filename = 'result_'+str(T.shape[0])+'x'+str(T.shape[1])+'_'+str(t)+'s.npy'
np.save(os.path.join('checkpoints', 'FDM_explicit', filename), T)

rows = np.linspace(0, m-1, 300).astype(int)
ulite = T[rows,:]

tdata = np.linspace(0, t, 300)
xdata = np.linspace(0, sum(l), 300)
tdata, xdata = np.meshgrid(tdata, xdata)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(tdata, xdata, ulite, cmap=cm.coolwarm)
ax.view_init(elev=25, azim=135)
plt.savefig('checkpoints/FDM_explicit/3d.eps')
#plt.show()