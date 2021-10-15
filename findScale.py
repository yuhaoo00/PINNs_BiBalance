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
        os.system("python main.py --expt 60 --find_scale --tag find --no_cuda")
        losses = np.load('checkpoints/PINNfind_60s_1e/losses_all.npy')
        losses_log.append(losses[0])
    
    losses_log = np.array(losses_log)
    np.save('checkpoints/PINNfind_60s_1e/losses_log.npy', losses_log)
else:
    losses_log = np.load('checkpoints/PINNfind_60s_1e/losses_log.npy')

losses_mean = np.mean(losses_log, axis=0)
print(losses_mean[c1])
print(losses_mean[c2])
print(losses_mean[c3])
c1range = np.array([min(losses_mean[c1]),max(losses_mean[c1])])
c2range = np.array([min(losses_mean[c2]),max(losses_mean[c2])])
c3range = np.array([min(losses_mean[c3]),max(losses_mean[c3])])
print(c1range,c2range,c3range)

print("[Before]")
print("IOU c1&c2: ",IOU(c1range,c2range))
print("IOU c1&c3: ",IOU(c1range,c3range))


labels1 = []
labels2 = []
labels3 = []
for n in c1:
    labels1.append(lossname[n])
for n in c2:
    labels2.append(lossname[n])
for n in c3:
    labels3.append(lossname[n])

fig = plt.figure(figsize=(18,5))
axe1=fig.add_subplot(131)
axe1.boxplot(losses_log[:,c1], labels=labels1, patch_artist=True, showfliers=False)
axe2=fig.add_subplot(132)
axe2.boxplot(losses_log[:,c2], labels=labels2, patch_artist=True, showfliers=False) 
axe3=fig.add_subplot(133)
axe3.boxplot(losses_log[:,c3], labels=labels3, patch_artist=True, showfliers=False) 
plt.show()


print("[After]")
c1range0 = c1range*(1e-2**2)
n2 = np.linspace(5e-5,5e-4,100)
n3 = np.linspace(1e-8,1e-7,100)
table = np.zeros((2,100))

for i in range(len(n2)):
    c2range0 = c2range*(n2[i]**2)
    table[0,i] = IOU(c1range0,c2range0)

for i in range(len(n3)):
    c3range0 = c3range*(n3[i]**2)
    table[1,i] = IOU(c1range0,c3range0)

print(table)
x,y = np.argmax(table, axis=1)
print(x,y)
print(n2[x],n3[y])

fig = plt.figure(figsize=(18,5))
ax1 = fig.add_subplot(121)
ax1.plot(n2,table[0])
ax2 = fig.add_subplot(122)
ax2.plot(n3,table[1])
plt.show()
#plt.savefig('eps/findscale.eps')