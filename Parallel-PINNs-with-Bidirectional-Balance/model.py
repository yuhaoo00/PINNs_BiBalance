import torch
import torch.nn as nn

def grad(outputs, inputs):
    """ compute the derivative of outputs associated with inputs

    Params
    ======
    outputs: (N, 1) tensor
    inputs: (N, D) tensor
    """
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               retain_graph=True,
                               create_graph=True)

def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['leaky_relu', 'LeakyReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus()
    else:
        raise ValueError(f'unknown activation function: {name}')

class DNN(nn.Module):
    """Deep Neural Network"""
    def __init__(self, dim_in, dim_out, dim_hidden, hidden_layers,
                 act_name='tanh', init_name=None):
        super().__init__()
        model = nn.Sequential()

        model.add_module('fc0', nn.Linear(dim_in, dim_hidden, bias=True))
        model.add_module('act0', activation(act_name))
        for i in range(1, hidden_layers):
            model.add_module(f'fc{i}', nn.Linear(dim_hidden, dim_hidden, bias=True))
            model.add_module(f'act{i}', activation(act_name))
        model.add_module(f'fc{hidden_layers}', nn.Linear(dim_hidden, dim_out, bias=True))

        self.model = model

        if init_name is not None:
            self.init_weight(init_name)

    def init_weight(self, name):
        if name == 'xavier_normal':
            nn_init = nn.init.xavier_normal_
        elif name == 'xavier_uniform':
            nn_init = nn.init.xavier_uniform_
        elif name == 'kaiming_normal':
            nn_init = nn.init.kaiming_normal_
        elif name == 'kaiming_uniform':
            nn_init = nn.init.kaiming_uniform_
        else:
            raise ValueError(f'unknown initialization function: {name}')

        for param in self.parameters():
            if len(param.shape) > 1:
                nn_init(param)

    def forward(self, x):
        x = self.model(x)
        return x

    def model_size(self):
        n_params = 0
        for param in self.parameters():
            n_params += param.numel()
        return n_params

class PINN(nn.Module):
    def __init__(self, problem, dim_in, dim_out, dim_hidden, hidden_layers,
                 act_name='tanh', init_name='xavier_normal'):
        super().__init__()
        
        self.problem = problem
        self.net1 = DNN(dim_in, dim_out, dim_hidden, hidden_layers, act_name=act_name, init_name=init_name)
        self.net2 = DNN(dim_in, dim_out, dim_hidden, hidden_layers, act_name=act_name, init_name=init_name)
        self.net3 = DNN(dim_in, dim_out, dim_hidden, hidden_layers, act_name=act_name, init_name=init_name)

    def forward(self, X, val=False):
        if val:
            y1 = self.net1(X[0])
            y2 = self.net2(X[1])
            y3 = self.net3(X[2])
            return torch.cat((y1, y2, y3), 0)

        # SHL
        X[0].requires_grad_(True)
        yshl = self.net1(X[0])
        grad_yshl = grad(yshl, X[0])[0]
        yshl_t = grad_yshl[:, [0]]
        yshl_x = grad_yshl[:, [1]]
        yshl_xx = grad(yshl_x, X[0])[0][:, [1]]
        r_shl = self.problem.r_shl(yshl_t, yshl_xx)

        X[1].requires_grad_(True)
        yshl_u0 = self.net1(X[1])
        u0_shl = self.problem.u0(yshl_u0)

        X[2].requires_grad_(True)
        yshl_ub0 = self.net1(X[2])
        yshl_ub0_x = grad(yshl_ub0, X[2])[0][:, [1]]
        ub_shl = self.problem.ub_shl(yshl_ub0, yshl_ub0_x)
        
        # MSR
        X[4].requires_grad_(True)
        ymsr = self.net2(X[4])
        grad_ymsr = grad(ymsr, X[4])[0]
        ymsr_t = grad_ymsr[:, [0]]
        ymsr_x = grad_ymsr[:, [1]]
        ymsr_xx = grad(ymsr_x, X[4])[0][:, [1]]
        r_msr = self.problem.r_msr(ymsr_t, ymsr_xx)

        X[5].requires_grad_(True)
        ymsr_u0 = self.net2(X[5])
        u0_msr = self.problem.u0(ymsr_u0)

        X[3].requires_grad_(True)
        yshl_ub1 = self.net1(X[3])
        yshl_ub1_x = grad(yshl_ub1, X[3])[0][:, [1]]
        X[6].requires_grad_(True)
        ymsr_ub0 = self.net2(X[6])
        ymsr_ub0_x = grad(ymsr_ub0, X[6])[0][:, [1]]
        ub_shlmsr1, ub_shlmsr2 = self.problem.ub_shlmsr(ymsr_ub0, yshl_ub1, ymsr_ub0_x, yshl_ub1_x)
        
        # LIN
        X[8].requires_grad_(True)
        ylin = self.net3(X[8])
        grad_ylin = grad(ylin, X[8])[0]
        ylin_t = grad_ylin[:, [0]]
        ylin_x = grad_ylin[:, [1]]
        ylin_xx = grad(ylin_x, X[8])[0][:, [1]]
        r_lin = self.problem.r_lin(ylin_t, ylin_xx)

        X[9].requires_grad_(True)
        ylin_u0 = self.net3(X[9])
        u0_lin = self.problem.u0(ylin_u0)

        X[7].requires_grad_(True)
        ymsr_ub1 = self.net2(X[7])
        ymsr_ub1_x = grad(ymsr_ub1, X[7])[0][:, [1]]
        X[10].requires_grad_(True)
        ylin_ub0 = self.net3(X[10])
        ylin_ub0_x = grad(ylin_ub0, X[10])[0][:, [1]]
        ub_msrlin1, ub_msrlin2 = self.problem.ub_msrlin(ylin_ub0, ymsr_ub1, ylin_ub0_x, ymsr_ub1_x)

        X[11].requires_grad_(True)
        ylin_ub1 = self.net3(X[11])
        ylin_ub1_x = grad(ylin_ub1, X[11])[0][:, [1]]
        ub_lin = self.problem.ub_linair(ylin_ub1, ylin_ub1_x)

        return [r_shl, u0_shl, ub_shl, \
                r_msr, u0_msr, ub_shlmsr1, ub_shlmsr2,\
                r_lin, u0_lin, ub_msrlin1, ub_msrlin2, ub_lin]

class SinglePINN(nn.Module):
    def __init__(self, problem, dim_in, dim_out, dim_hidden, hidden_layers,
                 act_name='tanh', init_name='xavier_normal'):
        super().__init__()
        
        self.problem = problem
        self.net = DNN(dim_in, dim_out, dim_hidden, hidden_layers, act_name=act_name, init_name=init_name)

    def forward(self, X, val=False):
        if val:
            y1 = self.net(X[0])
            y2 = self.net(X[1])
            y3 = self.net(X[2])
            return torch.cat((y1, y2, y3), 0)

        # SHL
        X[0].requires_grad_(True)
        yshl = self.net(X[0])
        grad_yshl = grad(yshl, X[0])[0]
        yshl_t = grad_yshl[:, [0]]
        yshl_x = grad_yshl[:, [1]]
        yshl_xx = grad(yshl_x, X[0])[0][:, [1]]
        r_shl = self.problem.r_shl(yshl_t, yshl_xx)

        X[1].requires_grad_(True)
        yshl_u0 = self.net(X[1])
        u0_shl = self.problem.u0(yshl_u0)

        X[2].requires_grad_(True)
        yshl_ub0 = self.net(X[2])
        yshl_ub0_x = grad(yshl_ub0, X[2])[0][:, [1]]
        ub_shl = self.problem.ub_shl(yshl_ub0, yshl_ub0_x)
        
        # MSR
        X[4].requires_grad_(True)
        ymsr = self.net(X[4])
        grad_ymsr = grad(ymsr, X[4])[0]
        ymsr_t = grad_ymsr[:, [0]]
        ymsr_x = grad_ymsr[:, [1]]
        ymsr_xx = grad(ymsr_x, X[4])[0][:, [1]]
        r_msr = self.problem.r_msr(ymsr_t, ymsr_xx)

        X[5].requires_grad_(True)
        ymsr_u0 = self.net(X[5])
        u0_msr = self.problem.u0(ymsr_u0)

        X[3].requires_grad_(True)
        yshl_ub1 = self.net(X[3])
        yshl_ub1_x = grad(yshl_ub1, X[3])[0][:, [1]]
        X[6].requires_grad_(True)
        ymsr_ub0 = self.net(X[6])
        ymsr_ub0_x = grad(ymsr_ub0, X[6])[0][:, [1]]
        ub_shlmsr1, ub_shlmsr2 = self.problem.ub_shlmsr(ymsr_ub0, yshl_ub1, ymsr_ub0_x, yshl_ub1_x)
        
        # LIN
        X[8].requires_grad_(True)
        ylin = self.net(X[8])
        grad_ylin = grad(ylin, X[8])[0]
        ylin_t = grad_ylin[:, [0]]
        ylin_x = grad_ylin[:, [1]]
        ylin_xx = grad(ylin_x, X[8])[0][:, [1]]
        r_lin = self.problem.r_lin(ylin_t, ylin_xx)

        X[9].requires_grad_(True)
        ylin_u0 = self.net(X[9])
        u0_lin = self.problem.u0(ylin_u0)

        X[7].requires_grad_(True)
        ymsr_ub1 = self.net(X[7])
        ymsr_ub1_x = grad(ymsr_ub1, X[7])[0][:, [1]]
        X[10].requires_grad_(True)
        ylin_ub0 = self.net(X[10])
        ylin_ub0_x = grad(ylin_ub0, X[10])[0][:, [1]]
        ub_msrlin1, ub_msrlin2 = self.problem.ub_msrlin(ylin_ub0, ymsr_ub1, ylin_ub0_x, ymsr_ub1_x)

        X[11].requires_grad_(True)
        ylin_ub1 = self.net(X[11])
        ylin_ub1_x = grad(ylin_ub1, X[11])[0][:, [1]]
        ub_lin = self.problem.ub_linair(ylin_ub1, ylin_ub1_x)

        return [r_shl, u0_shl, ub_shl, \
                r_msr, u0_msr, ub_shlmsr1, ub_shlmsr2,\
                r_lin, u0_lin, ub_msrlin1, ub_msrlin2, ub_lin]