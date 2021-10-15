import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
train = False

def IOU(l1,l2):
    if l2[0]>=l1[1] or l1[0]>=l2[1]:
        return 0
    jiao = min(l1[1],l2[1]) - max(l1[0],l2[0])
    bing = max(l1[1],l2[1]) - min(l1[0],l2[0])
    return jiao/bing

lossname = ['r_shl', 'u0_shl', 'ub_shl', \
            'r_msr', 'u0_msr', 'ub_shlmsr1', 'ub_shlmsr2',\
            'r_lin', 'u0_lin', 'ub_msrlin1', 'ub_msrlin2', 'ub_lin']
c1 = [1,4,8,5,9]
c2 = [2,6,10,11]
c3 = [0,3,7]

if train:
    nexp = 50
    losses_log = []

    for i in tqdm(range(nexp)):
        os.system("python main.py --expt 60 --find_scale --tag findAfter --no_cuda --findAfter")
        losses = np.load('checkpoints/PINNfindAfter_60s_1e/losses_all.npy')
        losses_log.append(losses[0])
    
    losses_log = np.array(losses_log)
    np.save('checkpoints/PINNfindAfter_60s_1e/losses_log.npy', losses_log)
else:
    losses_log = np.load('checkpoints/PINNfindAfter_60s_1e/losses_log.npy')

losses_mean = np.mean(losses_log, axis=0)
print(losses_mean[c1])
print(losses_mean[c2])
print(losses_mean[c3])
c1range = np.array([min(losses_mean[c1]),max(losses_mean[c1])])
c2range = np.array([min(losses_mean[c2]),max(losses_mean[c2])])
c3range = np.array([min(losses_mean[c3]),max(losses_mean[c3])])
print(c1range,c2range,c3range)
print("IOU c1&c2: ",IOU(c1range,c2range))
print("IOU c1&c3: ",IOU(c1range,c3range))

#losses_log = losses_log.T

losses_log = losses_log[:,c1+c2+c3]
labels = []
for n in c1+c2+c3:
    labels.append(lossname[n])
print(losses_log.shape)

figure,axes=plt.subplots()
axes.boxplot(losses_log, labels=labels, patch_artist=True, showfliers=False) 
plt.show()
