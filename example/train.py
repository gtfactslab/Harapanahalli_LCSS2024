import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from invariance import NeuralNetwork, NeuralNetworkData, ScaledMSELoss
from invariance.utils import file_to_numpy

# MODELNAME = '100r100r2'
# FILENAMES = ['twoobs_20230302-061841', 
#              'twoobs_20230302-061907', 
#              'twoobs_20230302-062016', 
#              'twoobs_20230302-063704', 
#              'twoobs_20230302-073059']
MODELNAME = '100r100r2_MPC'
FILENAMES = ['mpc_10_025_20230903-110137', 
             'mpc_10_025_20230903-140258',
             'mpc_10_025_20230903-224021']

EPOCHS = 30
LEARNING = 1
LEARNING_GAM = 0.9
TRAINP = 0.9
LAYER1 = 100
LAYER2 = 100

device = 'cuda'

X, U = file_to_numpy(FILENAMES)
SPLIT_LOC = int(TRAINP * X.shape[0])

print(X.shape)

# Xtrain = X[:SPLIT_LOC,:]
# Utrain = U[:SPLIT_LOC,:]
# Xtest = X[SPLIT_LOC:,:]
# Utest = U[SPLIT_LOC:,:]
Xtrain = X[:SPLIT_LOC,:].astype(np.float32)
Utrain = U[:SPLIT_LOC,:].astype(np.float32)
Xtest = X[SPLIT_LOC:,:].astype(np.float32)
Utest = U[SPLIT_LOC:,:].astype(np.float32)

print(Xtrain.dtype)

fig, axs = plt.subplots(2, 2,figsize=[8,8], dpi=100)
axs = axs.reshape(-1)
axs[0].scatter(Xtest[:,0], Xtest[:,2], s=1)
axs[0].set_title("px vs py")
axs[0].set_xlim([-15,15]); axs[0].set_ylim([-15,15])
axs[1].scatter(Xtest[:,1], Xtest[:,3], s=1)
axs[1].set_title("vy vs vx")
axs[2].scatter(Utest[:,0], Utest[:,1], s=1)
axs[2].set_title("ux vs uy (data)")
plt.ion(); plt.show(); plt.pause(1)
# plt.draw()

Xtrain = torch.from_numpy(Xtrain).to(device)
Utrain = torch.from_numpy(Utrain).to(device)
Xtest = torch.from_numpy(Xtest).to(device)
Utest = torch.from_numpy(Utest).to(device)

train_data = NeuralNetworkData(Xtrain, Utrain)
test_data  = NeuralNetworkData(Xtest, Utest)
train_dl = DataLoader(train_data, batch_size=64, shuffle=True)
val_dl   = DataLoader(test_data, batch_size=256, shuffle=True)
test_dl  = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

print("Training set size: ", len(train_data))
print("Testing  set size: ", len(test_data))

net = NeuralNetwork('models/' + MODELNAME, False, device)
print(net)

learning_rate = LEARNING
loss = ScaledMSELoss(scale=train_data.maxU())
optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LEARNING_GAM)

Xt, Ut = next(iter(test_dl))
for epoch in range(EPOCHS) :
    pbar = tqdm(iter(train_dl))
    for Xi, Ui in pbar :
        l = loss(net(Xi), Ui)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        pbar.set_description(f'Epoch {epoch+1}, Loss={l:.8f}')
    scheduler.step()
    Upred = net(Xt)
    test_loss = loss(Upred, Ut)
    Upred = Upred.detach().cpu().numpy()
    axs[3].clear()
    axs[3].scatter(Upred[:,0], Upred[:,1], s=1)
    axs[3].set_xlim(axs[2].get_xlim()); axs[3].set_ylim(axs[2].get_ylim())
    axs[3].set_title("uang vs uacc (predicted)")
    fig.suptitle(f"Scaled RMSE on Testing Set: {torch.sqrt(test_loss)}.4")
    print(f'Epoch: {epoch+1}, RMSE={torch.sqrt(test_loss):8f}')
    plt.ion(); plt.show(); plt.pause(0.2)

print(f'Loss on Full Test Set: {test_loss}')

# savepath = 'models/' + MODELNAME + '.pt'
# print(f'Saving to {savepath}')
# torch.save(net.state_dict(), savepath)

net.save()

plt.ioff(); plt.show()
