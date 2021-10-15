import argparse
import torch
import shutil
import os
import matplotlib
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

from problem import *
from dataset import *
from model import *


import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn", lineno=0)
torch.set_printoptions(precision=16)
plt.rc('font',family='Times New Roman') 

parser = argparse.ArgumentParser()
parser.add_argument('--find_scale', action='store_true', default=False)
parser.add_argument('--no_cuda',    action='store_true', default=False)
parser.add_argument('--inference',  action='store_true', default=False)
parser.add_argument('--debug',      action='store_true', default=False)
parser.add_argument('--no_PSF',     action='store_true', default=False)
parser.add_argument('--no_FBM',     action='store_true', default=False)
parser.add_argument('--no_BBM',     action='store_true', default=False)
parser.add_argument('--eps',        action='store_true', default=False)
parser.add_argument('--findAfter',  action='store_true', default=False)

parser.add_argument('--resume', action='store_true', default=False, help='Continue training from an existing trained file')
parser.add_argument('--netpath', type=str, default=None, help='Trained network state file path')
parser.add_argument('--dim_hidden', type=int, default=10, help='neurons in hidden layers')
parser.add_argument('--hidden_layers', type=int, default=4, help='number of hidden layers')
parser.add_argument('--seed', type=int, default=None, help='random seed') #211

parser.add_argument('--tag', type=str, default='', help='personal tag, which can be any string')
parser.add_argument('--scales', type=list, default=[1e-2,1.27e-4,4.72e-8])

parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--epochs_Adam', type=int, default=1000, help='epochs for Adam optimizer')
parser.add_argument('--epochs_start', type=int, default=0, help='epochs for Start index')

parser.add_argument('--n_t', type=int, default=300, help='sample points in t-direction for uniform sample')
parser.add_argument('--n_x', type=list, default=[50,70,200], help='sample points in x-direction for uniform sample')
parser.add_argument('--expt', type=float, default=60, help='maximum experimental Time (s)')
args = parser.parse_args()


def model_path(model_name):
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    dirname = '{}{}_{}s_{}e{}'.format(model_name, args.tag, int(args.expt), args.epochs_Adam, '_Stiff' if args.stiff else '')
    path = os.path.join('checkpoints', dirname)
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def save_model(state, is_best=None, save_dir=None):
    torch.save(state, os.path.join(save_dir, 'last_model.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(save_dir, 'last_model.pth.tar'), os.path.join(save_dir, 'best_model.pth.tar'))

class Trainer(object):
    def __init__(self, args):
        self.debug = args.debug
        self.find_scale = args.find_scale
        self.stiff = args.stiff
        self.no_FBM = args.no_FBM
        self.eps = args.eps
        self.device = args.device
        self.layers = args.hidden_layers
        self.problem = args.problem

        self.criterion = nn.MSELoss()
        self.lossnum = 12
        self.lossname = ['r_shl', 'u0_shl', 'ub_shl', \
                         'r_msr', 'u0_msr', 'ub_shlmsr1', 'ub_shlmsr2',\
                         'r_lin', 'u0_lin', 'ub_msrlin1', 'ub_msrlin2', 'ub_lin']
        net1_loss = [0,1,2,5,6]
        net2_loss = [3,4,5,6,9,10]
        net3_loss = [7,8,9,10,11]
        self.nets = [net1_loss, net2_loss, net3_loss]
        self.lam = torch.ones(self.lossnum).float().to(self.device)
        self.loss_all = []
        self.losses_all =[]
        self.loss_val_all = []

        self.model = args.model

        self.epochs_start = args.epochs_start
        self.epochs_Adam = args.epochs_Adam
        self.optimizer_Adam = optim.Adam(self.model.parameters(), lr=args.lr)

        self.model_path = model_path(self.model.__class__.__name__)
        self.model.to(self.device)
        self.model.zero_grad()

        self.X = args.trainset(device=self.device)
        self.X_val, _, _ = args.validset(device=self.device)

        self.grad_dict = {}
        for i in range(3):
            self.grad_dict['net{}'.format(i+1)] = [[[] for _ in range(len(self.nets[i]))] for _ in range(self.layers)]

        self.u_real = np.load(f'checkpoints/FDM_explicit/result_320x300_{int(args.expt)}s.npy')/1e3
        self.u_real = torch.from_numpy(self.u_real).to(self.device)
        self.u_real = torch.flatten(self.u_real)

    def train(self):
        best_loss = 1.e10

        for epoch in range(self.epochs_start, self.epochs_Adam):
            if self.stiff:
                if  epoch == 4000 or epoch == 12000:
                    self.problem.stiff_scale()
                    print(self.problem.scale)

            loss, losses = self.train_Adam()
            self.loss_all.append(loss)
            self.losses_all.append(losses)
            if (epoch + 1) % 100 == 0:
                self.infos_Adam(epoch + 1, loss, losses)
                
                valid_loss = self.validate(epoch)
                self.loss_val_all.append(valid_loss)
                is_best = valid_loss < best_loss
                best_loss = valid_loss if is_best else best_loss
                state = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'best_loss': best_loss,
                }
                save_model(state, is_best, save_dir=self.model_path)
            if self.debug and (epoch + 1) % 250 == 0:
                self.check_grad(epoch, plot=True)

        self.check_gradstd(self.epochs_Adam-1)
        
        return
    
    def LOSS(self, preds):
        losses = torch.zeros((self.lossnum,1)).to(self.device)
        for i in range(0, self.lossnum):
            losses[i] = self.criterion(preds[i], torch.zeros_like(preds[i]).to(self.device))
        loss = torch.matmul(self.lam, losses)
        return loss, losses.squeeze()

    def train_Adam(self):
        self.optimizer_Adam.zero_grad()

        preds = self.model(self.X)
        loss, losses = self.LOSS(preds)
        loss.backward()
        self.optimizer_Adam.step()

        return loss.item(), losses.detach().cpu().numpy()

    def infos_Adam(self, epoch, loss, losses):
        lam_np = self.lam.detach().cpu().numpy()
        lossinfo = ' '
        for i in range(0, self.lossnum):
            lossinfo += f'<{losses[i]:.2e}> '
            
        infos = 'Adam  ' + \
                f'Epoch #{epoch:5d}/{self.epochs_Adam} ' + \
                f'| Loss: {loss:.4e} = ' + lossinfo
        print(infos)

    def validate(self, epoch):
        self.model.eval()

        u_pred = self.model(self.X_val, val=True)
        u_pred = u_pred.detach().squeeze()
        if self.no_FBM:
            u_pred = u_pred/1e3
        loss = self.criterion(u_pred, self.u_real)
        infos = 'Valid ' + \
                f'Epoch #{epoch + 1:5d}/{self.epochs_Adam} ' + \
                f'| Loss: {loss.item():.4e} '
        print(infos)

        self.model.train()
        return loss.item()
    
    def check_grad(self, epoch, plot=False):
        preds = self.model(self.X)
        loss, losses = self.LOSS(preds)
        loss.backward(retain_graph=True)

        if plot:
            cnt = 1
            fig = plt.figure(figsize=(13, 8))

        for k in range(len(self.nets)):
            for i in range(self.layers):
                if plot:
                    ax = plt.subplot(3, 4, cnt)
                    if k==0:
                        ax.set_title('Layer {}'.format(i+1))
                    ax.set_yscale('symlog')
                
                for j in range(len(self.nets[k])):
                    lossj  = losses[self.nets[k][j]]
                    nowgrad = torch.autograd.grad(lossj,
                                                eval('self.model.net'+str(k+1)).model[2*i].weight,
                                                grad_outputs=torch.ones_like(lossj),
                                                retain_graph=True)[0]
                    nowgrad = torch.flatten(nowgrad).detach()
                    self.grad_dict['net{}'.format(k+1)][i][j].append(torch.std(nowgrad).item())
                    
                    if plot:
                        nowgrad = nowgrad.cpu().numpy()
                        sns.distplot(nowgrad, hist=False,
                                    kde_kws={"shade": False},
                                    norm_hist=True, label='loss_{}'.format(self.lossname[self.nets[k][j]]))
                
                if plot:
                    if i == self.layers-1:
                        handles, labels = ax.get_legend_handles_labels()
                        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05,1))
                    else:
                        ax.legend().remove()
                    ax.set_xlim([-3.0, 3.0])
                    ax.set_ylim([0,1000])
                    ax.set(ylabel='')
                    cnt += 1
        
        if plot:
            plt.tight_layout()
            figname = str(epoch)+ ('.png' if not self.eps else '.eps')
            filedir = os.path.join(self.model_path,'grad')
            if not os.path.exists(filedir):
                os.mkdir(filedir)
            plt.savefig(os.path.join(filedir, figname))
            plt.close()
        
        return losses.detach().cpu().numpy()

    def check_gradstd(self, epoch, save=True):
        data1 = np.array(self.grad_dict['net1'])
        data2 = np.array(self.grad_dict['net2'])
        data3 = np.array(self.grad_dict['net3'])
        dataall = [data1, data2, data3]
        x = np.linspace(0, epoch, data1.shape[2])

        cnt = 1
        fig = plt.figure(figsize=(13, 8))
        for k in range(len(self.nets)):
            datak = dataall[k]
            neti = self.nets[k]
            for i in range(self.layers):
                ax = plt.subplot(3, 4, cnt)
                if k==0:
                    ax.set_title('Layer {}'.format(i+1))
                for j in range(len(neti)):
                    ax.plot(x, datak[i,j,:], label='loss_{}'.format(self.lossname[neti[j]]))
                
                if i == self.layers-1:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05,1))
                #ax.set_xlim([-3.0, 3.0])
                ax.set_ylim([0, 1])
                cnt += 1
        
        plt.tight_layout()
        if save:
            filedir = os.path.join(self.model_path, 'grad')
            if not os.path.exists(filedir):
                os.mkdir(filedir)
            figname = 'gradstd'+('.png' if not self.eps else '.eps')
            plt.savefig(os.path.join(filedir, figname))
            plt.close()
        else:
            plt.show()
    
    def save_log(self):
        np.save(os.path.join(self.model_path, 'loss_all.npy'), np.array(self.loss_all))
        np.save(os.path.join(self.model_path, 'loss_val_all.npy'), np.array(self.loss_val_all))
        if self.find_scale:
             np.save(os.path.join(self.model_path, 'losses_all.npy'), np.array(self.losses_all))

class Tester(object):
    def __init__(self, args):
        self.device = args.device
        self.problem = args.problem
        self.criterion = nn.MSELoss()
        self.model = args.model
        self.no_FBM = args.no_FBM
        self.eps = args.eps

        if not args.inference:
            self.model_path = model_path(self.model.__class__.__name__)
            modelstate = torch.load(os.path.join(self.model_path, 'best_model.pth.tar'))
            self.model.load_state_dict(modelstate['state_dict'])
        else:
            strlist = args.netpath.split('\\')
            self.model_path = '\\'.join(strlist[:-1])
        self.model.to(self.device)

        self.X, self.t, self.x = args.testset(device=self.device)

        self.u_real = np.load(f'checkpoints/FDM_explicit/result_320x300_{int(args.expt)}s.npy')/1e3
        self.u_real = self.u_real.reshape(self.t.shape)
    
    def mycriterion(self, pred, real):
            # Calculate the MSE between Pred value and Real value
            real = torch.flatten(torch.from_numpy(real).to(self.device))
            pred = torch.flatten(torch.from_numpy(pred).to(self.device))
            loss = self.criterion(pred, real)
            return loss

    def predict(self, save=True):
        self.model.eval()
        u_pred = self.model(self.X, val=True)
        u_pred = u_pred.detach().cpu().numpy()
        u_pred = u_pred.reshape(self.t.shape)
        if self.no_FBM:
            u_pred = u_pred/1e3

        u1_pred = u_pred[:50,:]
        u2_pred = u_pred[50:120,:]
        u3_pred = u_pred[120:320,:]
        u1_real = self.u_real[:50,:]
        u2_real = self.u_real[50:120,:]
        u3_real = self.u_real[120:320,:]

        loss1 = self.mycriterion(u1_pred, u1_real)
        loss2 = self.mycriterion(u2_pred, u2_real)
        loss3 = self.mycriterion(u3_pred, u3_real)
        loss = self.mycriterion(u_pred, self.u_real)

        print(f"\nThe MSE between PRED & REAL value is <<{loss.item():4e}>>")
        print(f"SHL<{loss1.item():4e}> MSR<{loss2.item():4e}> LIN<{loss3.item():4e}>")

        # Fig 1 vs.png
        fig = plt.figure(figsize=(18,4))
        norm1 = matplotlib.colors.Normalize(vmin=0.25, vmax=np.max(u_pred))
        norm2 = matplotlib.colors.Normalize(vmin=-0.1, vmax=0.1)
        ax1 = fig.add_subplot(131)
        ax1.set_title('Groud Truth')
        ax1.set_xlabel('t (s)')
        ax1.set_ylabel('x (mm)')
        a1 = ax1.pcolormesh(self.t, self.x, self.u_real, cmap='rainbow',shading='auto', norm=norm1, rasterized=True)
        ax2 = fig.add_subplot(132)
        ax2.set_title('Pred T')
        ax2.set_xlabel('t (s)')
        a2 = ax2.pcolormesh(self.t, self.x, u_pred, cmap='rainbow',shading='auto', norm=norm1, rasterized=True)
        fig.colorbar(a1, ax=[ax1, ax2])
        ax3 = fig.add_subplot(133)
        ax3.set_title('Error')
        ax3.set_xlabel('t (s)')
        ax3.set_ylabel('x (mm)')
        a3 = ax3.pcolormesh(self.t, self.x, self.u_real-u_pred, cmap='coolwarm',shading='auto',norm=norm2, rasterized=True)
        fig.colorbar(a3, ax=[ax3])
        if self.eps:
            plt.savefig(os.path.join(self.model_path, 'vs.eps'), bbox_inches='tight')
        plt.savefig(os.path.join(self.model_path, 'vs.png'), bbox_inches='tight')
        plt.close()

        # Fig 2 3Dview
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.t, self.x, u_pred, cmap=cm.coolwarm)
        ax.view_init(elev=25, azim=135)
        if self.eps:
            plt.savefig(os.path.join(self.model_path, '3d.eps'))
        else:
            plt.show()

        if save:
            np.save(os.path.join(self.model_path, 'predu.npy'), u_pred)

if __name__ == "__main__":
    if not args.no_cuda:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device("cpu")
    
    if args.eps:
        matplotlib.use('Agg')

    '''FBM/BBM Mode'''
    Lfab = np.array([0.6,1.45,5.05])
    T0 = 310.15*1e-3
    Tg = 2000*1e-3
    if args.no_BBM:
        args.scales = [1,1,1]
    if args.no_FBM:
        Lfab = Lfab*1e-3
        T0 = T0*1e3
        Tg = Tg*1e3

    '''FindScale Mode'''
    if args.find_scale:
        args.epochs_Adam = 1
        args.epochs_start = 0
        args.seed = None
        args.debug = False
        if not args.findAfter:
            args.scales = [1,1,1]
    else:
        print(f'device {args.device}')

    '''Build DataSet and Main Model'''
    if args.seed:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    # build problem by FBM/BBM
    args.problem = Problem([300*1377,862*2100,74.2*1726], [0.082,0.37,0.045],\
                            T0, Tg, [40,8.496], 0.02, 0.01,\
                            np.array([[0,args.expt],[0,Lfab[0]],[Lfab[0]+0.001,Lfab[1]],[Lfab[1]+0.001,Lfab[2]]]),\
                            args.no_FBM)
    args.problem.set_scale(args.scales)
    
    
    args.trainset = Trainset(args.problem, args.n_t, args.n_x)
    args.validset = Testset(args.problem, args.n_t, args.n_x)
    args.testset = Testset(args.problem, args.n_t, args.n_x)

    if args.no_PSF:
        args.model = SinglePINN(args.problem, 2, 1,\
                    dim_hidden=args.dim_hidden, hidden_layers=args.hidden_layers,\
                    act_name='tanh', init_name='kaiming_normal')
    else:
        args.model = PINN(args.problem, 2, 1,\
                    dim_hidden=args.dim_hidden, hidden_layers=args.hidden_layers,\
                    act_name='tanh', init_name='kaiming_normal')

    '''Train or Test'''
    if args.resume or args.inference:
        # load netpath if resume/inference mode
        if args.netpath and os.path.exists(args.netpath):
            modelstate = torch.load(args.netpath)
            args.model.load_state_dict(modelstate['state_dict'])
            args.epochs_start = modelstate['epoch']+1
            print("["+args.netpath+"] The file has been loaded!")
        else:
            print("Enter the trained network state file path!")
            exit(0)

    if not args.inference:
        # train(or resume)
        trainer = Trainer(args)
        trainer.train()
        trainer.save_log()
    else:
        # test/inference
        tester = Tester(args)
        tester.predict()