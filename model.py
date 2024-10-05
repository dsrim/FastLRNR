from re import T
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb
from config import get_config


class LR_PINN_phase1(nn.Module):
    def __init__(self, hidden_dim, reg_param=1e-10, alpha_dim=20):
        super(LR_PINN_phase1, self).__init__()

        self.start_layer = nn.Linear(2, hidden_dim)
        self.end_layer = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim
        self.alpha_dim = alpha_dim
        scale = 1/hidden_dim
        self.scale = scale
        
        self.col_basis_0 = nn.Parameter(scale*torch.rand(hidden_dim, alpha_dim))
        self.col_basis_1 = nn.Parameter(scale*torch.rand(hidden_dim, alpha_dim))
        self.col_basis_2 = nn.Parameter(scale*torch.rand(hidden_dim, alpha_dim))

        self.row_basis_0 = nn.Parameter(scale*torch.rand(hidden_dim, alpha_dim))
        self.row_basis_1 = nn.Parameter(scale*torch.rand(hidden_dim, alpha_dim))
        self.row_basis_2 = nn.Parameter(scale*torch.rand(hidden_dim, alpha_dim))
        
        self.meta_layer_1 = nn.Linear(3, alpha_dim)
        self.meta_layer_2 = nn.Linear(alpha_dim, alpha_dim)
        self.meta_layer_3 = nn.Linear(alpha_dim, alpha_dim)
        
        self.meta_alpha_0 = nn.Linear(alpha_dim, alpha_dim)
        self.meta_alpha_1 = nn.Linear(alpha_dim, alpha_dim)
        self.meta_alpha_2 = nn.Linear(alpha_dim, alpha_dim)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.reg_param = nn.Parameter(torch.tensor([reg_param]))
        self.reg_param.requires_grad = False

    def lrmatmul(self, col, alpha, row, emb_out):

        emb_out = torch.einsum("ij,...i",col, emb_out)
        emb_out = torch.einsum("...j,...j->...j",alpha, emb_out)
        emb_out = torch.einsum("ij,...j",row, emb_out)
        return emb_out
        
    def forward(self, x, t, beta, nu, rho):

        lrmatmul = self.lrmatmul
        ##### meta learning #####
        meta_input = torch.cat([beta, nu, rho], dim=1)
        meta_output = self.meta_layer_1(meta_input)
        meta_output = self.tanh(meta_output)
        meta_output = self.meta_layer_2(meta_output)
        meta_output = self.tanh(meta_output)
        meta_output = self.meta_layer_3(meta_output)
        meta_output = self.tanh(meta_output)

        alpha_0 = self.relu(self.meta_alpha_0(meta_output))
        alpha_1 = self.relu(self.meta_alpha_1(meta_output))
        alpha_2 = self.relu(self.meta_alpha_2(meta_output))

        self.meta_alpha_0_output = alpha_0
        self.meta_alpha_1_output = alpha_1
        self.meta_alpha_2_output = alpha_2

        ##### main neural network #####
        inputs = torch.cat([x, t], axis=1)

        emb_out = self.start_layer(inputs)
        emb_out = self.tanh(emb_out)

        emb_out = lrmatmul(self.col_basis_0, alpha_0, self.row_basis_0, emb_out)
        emb_out = self.tanh(emb_out)
        emb_out = lrmatmul(self.col_basis_1, alpha_1, self.row_basis_1, emb_out)
        emb_out = self.tanh(emb_out)
        emb_out = lrmatmul(self.col_basis_2, alpha_2, self.row_basis_2, emb_out)
        emb_out = self.tanh(emb_out)
        
        emb_out = emb_out.unsqueeze(dim=1)
        emb_out = self.end_layer(emb_out)
        emb_out = emb_out.squeeze(dim=1)
        
        return emb_out, self.col_basis_0, self.col_basis_1, self.col_basis_2, self.row_basis_0, self.row_basis_1, self.row_basis_2


    
    

class LR_PINN_phase2(nn.Module):
    def __init__(self, hidden_dim, start_w, start_b, end_w, end_b,
                 col_0, col_1, col_2, row_0, row_1, row_2, 
                 alpha_0, alpha_1, alpha_2):
        
        super(LR_PINN_phase2, self).__init__()

        self.start_layer = nn.Linear(2, hidden_dim)
        self.end_layer = nn.Linear(hidden_dim, 1)
        
        self.start_layer.weight = nn.Parameter(start_w, requires_grad=False)
        self.start_layer.bias = nn.Parameter(start_b, requires_grad=False)
        self.end_layer.weight = nn.Parameter(end_w, requires_grad=False)
        self.end_layer.bias = nn.Parameter(end_b, requires_grad=False)
        
        self.hidden_dim = hidden_dim
        self.scale = 1/hidden_dim
        
        self.col_basis_0 = nn.Parameter(col_0, requires_grad=False)
        self.col_basis_1 = nn.Parameter(col_1, requires_grad=False)
        self.col_basis_2 = nn.Parameter(col_2, requires_grad=False)

        self.row_basis_0 = nn.Parameter(row_0, requires_grad=False)
        self.row_basis_1 = nn.Parameter(row_1, requires_grad=False)
        self.row_basis_2 = nn.Parameter(row_2, requires_grad=False)
    
        self.alpha_0 = nn.Parameter(alpha_0)
        self.alpha_1 = nn.Parameter(alpha_1)
        self.alpha_2 = nn.Parameter(alpha_2)

        self.tanh = nn.Tanh()

    def forward(self, x, t):
        
        weight_0 = torch.matmul(torch.matmul(self.col_basis_0, torch.diag(self.alpha_0)), self.row_basis_0.T)
        weight_1 = torch.matmul(torch.matmul(self.col_basis_1, torch.diag(self.alpha_1)), self.row_basis_1.T)
        weight_2 = torch.matmul(torch.matmul(self.col_basis_2, torch.diag(self.alpha_2)), self.row_basis_2.T)

        ##### main neural network #####
        inputs = torch.cat([x, t], axis=1)
        emb_out = self.start_layer(inputs)
        emb_out = self.tanh(emb_out)
        
        emb_out = torch.matmul(emb_out, weight_0)
        emb_out = self.tanh(emb_out)
        
        emb_out = torch.matmul(emb_out, weight_1)
        emb_out = self.tanh(emb_out)
        
        emb_out = torch.matmul(emb_out, weight_2)
        emb_out = self.tanh(emb_out)
        
        emb_out = self.end_layer(emb_out)
        return emb_out


class LR_PINN_phase_fast(nn.Module):
    def __init__(self, hidden_dim, start_w, start_b, end_w, end_b,
                 col_0, col_1, col_2, row_0, row_1, row_2, 
                 alpha_0, alpha_1, alpha_2, row_index, eim_basis, r):

        super(LR_PINN_phase_fast, self).__init__()

        self.r = r

        row_index = row_index[:, :r]
        eim_basis = eim_basis[:, :, :r]

        self.row_index = row_index
        self.eim_basis = eim_basis

        eim_square_inv = torch.zeros((4, r, r))
        for i in range(4):
            inv_square = torch.linalg.solve(eim_basis[i, row_index[i, :], :].T, torch.eye(r))
            eim_square_inv[i, :, :] = inv_square

        self.eim_square_inv = nn.Parameter(eim_square_inv, requires_grad=False)

        self.start_layer = nn.Linear(2, r)
        self.end_layer = nn.Linear(r, 1)

        start_w_sm = start_w[row_index[0, :], :]
        start_b_sm = start_b[row_index[0, :]]

        end_w_sm = torch.matmul(end_w, eim_basis[3, ...])
        end_b_sm = end_b

        self.start_layer.weight = nn.Parameter(start_w_sm, requires_grad=False)
        self.start_layer.bias = nn.Parameter(start_b_sm, requires_grad=False)

        self.end_layer.weight = nn.Parameter(end_w_sm, requires_grad=False)
        self.end_layer.bias = nn.Parameter(end_b_sm, requires_grad=False)
        
        self.hidden_dim = hidden_dim
        self.scale = 1/hidden_dim
        
        self.tanh = nn.Tanh()

        row_0_sm = row_0[row_index[1, :], :]
        row_1_sm = row_1[row_index[2, :], :]
        row_2_sm = row_2[row_index[3, :], :]

        col_0_sm = torch.matmul(eim_basis[0, :, :].T, col_0)
        col_1_sm = torch.matmul(eim_basis[1, :, :].T, col_1)
        col_2_sm = torch.matmul(eim_basis[2, :, :].T, col_2)

        self.col_basis_0 = nn.Parameter(col_0_sm, requires_grad=False)
        self.col_basis_1 = nn.Parameter(col_1_sm, requires_grad=False)
        self.col_basis_2 = nn.Parameter(col_2_sm, requires_grad=False)

        self.row_basis_0 = nn.Parameter(row_0_sm, requires_grad=False)
        self.row_basis_1 = nn.Parameter(row_1_sm, requires_grad=False)
        self.row_basis_2 = nn.Parameter(row_2_sm, requires_grad=False)

        self.alpha_0 = nn.Parameter(alpha_0)
        self.alpha_1 = nn.Parameter(alpha_1)
        self.alpha_2 = nn.Parameter(alpha_2)


    def forward(self, x, t):

        weight_0 = torch.matmul(torch.matmul(self.col_basis_0, torch.diag(self.alpha_0)), self.row_basis_0.T)
        weight_1 = torch.matmul(torch.matmul(self.col_basis_1, torch.diag(self.alpha_1)), self.row_basis_1.T)
        weight_2 = torch.matmul(torch.matmul(self.col_basis_2, torch.diag(self.alpha_2)), self.row_basis_2.T)

        eim_square_inv = self.eim_square_inv

        ##### main neural network #####
        inputs = torch.cat([x, t], axis=1)
        emb_out = self.start_layer(inputs)
        emb_out = self.tanh(emb_out)
        emb_out = torch.matmul(emb_out, eim_square_inv[0, ...])

        emb_out = torch.matmul(emb_out, weight_0)
        emb_out = self.tanh(emb_out)
        emb_out = torch.matmul(emb_out, eim_square_inv[1, ...])

        emb_out = torch.matmul(emb_out, weight_1)
        emb_out = self.tanh(emb_out)
        emb_out = torch.matmul(emb_out, eim_square_inv[2, ...])

        emb_out = torch.matmul(emb_out, weight_2)
        emb_out = self.tanh(emb_out)
        emb_out = torch.matmul(emb_out, eim_square_inv[3, ...])
        
        emb_out = self.end_layer(emb_out)
        return emb_out


class LR_PINN_phase_ortho(nn.Module):
    def __init__(self, hidden_dim, start_w, start_b, end_w, end_b,
                 col_0, col_1, col_2, row_0, row_1, row_2, 
                 alpha_0, alpha_1, alpha_2):
        
        super(LR_PINN_phase_ortho, self).__init__()

        self.start_layer = nn.Linear(2, hidden_dim)
        self.end_layer = nn.Linear(hidden_dim, 1)
        
        self.start_layer.weight = nn.Parameter(start_w)
        self.start_layer.bias = nn.Parameter(start_b)
        self.end_layer.weight = nn.Parameter(end_w)
        self.end_layer.bias = nn.Parameter(end_b)
        
        self.hidden_dim = hidden_dim
        self.scale = 1/hidden_dim
        
        self.col_basis_0 = nn.Parameter(col_0, requires_grad=False)
        self.col_basis_1 = nn.Parameter(col_1, requires_grad=False)
        self.col_basis_2 = nn.Parameter(col_2, requires_grad=False)

        self.row_basis_0 = nn.Parameter(row_0, requires_grad=False)
        self.row_basis_1 = nn.Parameter(row_1, requires_grad=False)
        self.row_basis_2 = nn.Parameter(row_2, requires_grad=False)
    
        self.alpha_0 = nn.Parameter(alpha_0)
        self.alpha_1 = nn.Parameter(alpha_1)
        self.alpha_2 = nn.Parameter(alpha_2)

        self.tanh = nn.Tanh()

    def forward(self, x, t):
        
        weight_0 = torch.matmul(torch.matmul(self.col_basis_0, torch.diag(self.alpha_0)), self.row_basis_0)
        weight_1 = torch.matmul(torch.matmul(self.col_basis_1, torch.diag(self.alpha_1)), self.row_basis_1)
        weight_2 = torch.matmul(torch.matmul(self.col_basis_2, torch.diag(self.alpha_2)), self.row_basis_2)

        ##### main neural network #####
        inputs = torch.cat([x, t], axis=1)
        emb_out = self.start_layer(inputs)
        emb_out = self.tanh(emb_out)

        self.out0 = emb_out.to("cpu").clone().detach().T
        
        emb_out = torch.matmul(emb_out, weight_0)
        emb_out = self.tanh(emb_out)

        self.out1 = emb_out.to("cpu").clone().detach().T
        
        emb_out = torch.matmul(emb_out, weight_1)
        emb_out = self.tanh(emb_out)

        self.out2 = emb_out.to("cpu").clone().detach().T
        
        emb_out = torch.matmul(emb_out, weight_2)
        emb_out = self.tanh(emb_out)

        self.out3 = emb_out.to("cpu").clone().detach().T
        
        emb_out = self.end_layer(emb_out)
        return emb_out



class LR_PINN_phase2_midout(nn.Module):
    def __init__(self, hidden_dim, start_w, start_b, end_w, end_b,
                 col_0, col_1, col_2, row_0, row_1, row_2, 
                 alpha_0, alpha_1, alpha_2):
        
        super(LR_PINN_phase2_midout, self).__init__()

        self.start_layer = nn.Linear(2, hidden_dim)
        self.end_layer = nn.Linear(hidden_dim, 1)
        
        self.start_layer.weight = nn.Parameter(start_w, requires_grad=False)
        self.start_layer.bias = nn.Parameter(start_b, requires_grad=False)
        self.end_layer.weight = nn.Parameter(end_w, requires_grad=False)
        self.end_layer.bias = nn.Parameter(end_b, requires_grad=False)
        
        self.hidden_dim = hidden_dim
        
        self.col_basis_0 = nn.Parameter(col_0, requires_grad=False)
        self.col_basis_1 = nn.Parameter(col_1, requires_grad=False)
        self.col_basis_2 = nn.Parameter(col_2, requires_grad=False)

        self.row_basis_0 = nn.Parameter(row_0, requires_grad=False)
        self.row_basis_1 = nn.Parameter(row_1, requires_grad=False)
        self.row_basis_2 = nn.Parameter(row_2, requires_grad=False)
    
        self.alpha_0 = nn.Parameter(alpha_0)
        self.alpha_1 = nn.Parameter(alpha_1)
        self.alpha_2 = nn.Parameter(alpha_2)

        self.tanh = nn.Tanh()


    def forward(self, x, t):
        
        coeff_0 = torch.matmul(self.col_basis_0, torch.diag(self.alpha_0))
        coeff_1 = torch.matmul(self.col_basis_1, torch.diag(self.alpha_1))
        coeff_2 = torch.matmul(self.col_basis_2, torch.diag(self.alpha_2))

        ##### main neural network #####
        inputs = torch.cat([x, t], axis=1)
        emb_out = self.start_layer(inputs)
        emb_out = self.tanh(emb_out)

        self.emb_out_0 = emb_out.detach().clone()

        emb_out = torch.matmul(emb_out, coeff_0)
        self.coeff_0 = emb_out.detach().clone()

        emb_out = torch.matmul(emb_out, self.row_basis_0.T)
        emb_out = self.tanh(emb_out)

        self.emb_out_1 = emb_out.detach().clone()

        emb_out = torch.matmul(emb_out, coeff_1)
        self.coeff_1 = emb_out.detach().clone()

        emb_out = torch.matmul(emb_out, self.row_basis_1.T)
        emb_out = self.tanh(emb_out)

        self.emb_out_2 = emb_out.detach().clone()

        emb_out = torch.matmul(emb_out, coeff_2)

        self.coeff_2 = emb_out.detach().clone()

        emb_out = torch.matmul(emb_out, self.row_basis_2.T)
        emb_out = self.tanh(emb_out)

        self.emb_out_3 = emb_out.detach().clone()

        emb_out = self.end_layer(emb_out)

        return emb_out
