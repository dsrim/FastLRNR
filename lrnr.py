
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from config import get_config
import torch
import time
import torch.optim as optim
import random
from data_gen import systems
import torch.backends.cudnn as cudnn
import pandas as pd
from model import LR_PINN_phase1, LR_PINN_phase2, LR_PINN_phase_fast, LR_PINN_phase2_midout
from utils import orthogonality_reg, f_cal_phase2, get_params, eim_row_index
import os, sys
from sklearn.metrics import explained_variance_score, max_error

# plotting
import matplotlib
matplotlib.rcParams["font.family"] = "monospace"
import matplotlib.pyplot as plt
#matplotlib.style.use("dark_background")

import numpy as np
import warnings


class LRNR():

    def __init__(self,
                 device = "cpu",
                 hidden_dim = 4000,
                 alpha_dim = 100,
                 epoch = 100,
                 load_epoch = 20000,
                 pde_type = "convection",
                 initial_condition = "sin_1",
                 start_coeff_1 = 1,
                 start_coeff_2 = 1,
                 start_coeff_3 = 1,
                 end_coeff_1 = 20,
                 end_coeff_2 = 20,
                 end_coeff_3 = 20,
                 target_coeff_1 = 10,
                 target_coeff_2 = 0,
                 target_coeff_3 = 0,
                 reg_param = 1e-6,
                 eim_rank = 50,
                 npts = 5,
                 learn_rate = 1e-5,
                 seed = 0,
                 n_ugrid = 256,
                 ):
        """

        Low Rank Neural Representation (LRNR) class
        
        """
        
        torch.manual_seed(0)
        np.random.seed(0)

        self.device = device
        self.hidden_dim = hidden_dim
        self.alpha_dim = alpha_dim
        self.epoch = epoch
        self.load_epoch = load_epoch
        self.pde_type = pde_type
        self.initial_condition = initial_condition

        # set coefficient range
        self.start_coeff_1 = start_coeff_1
        self.start_coeff_2 = start_coeff_2
        self.start_coeff_3 = start_coeff_3

        self.end_coeff_1 = end_coeff_1
        self.end_coeff_2 = end_coeff_2
        self.end_coeff_3 = end_coeff_3

        self.alpha_0_nz_index = None
        self.alpha_1_nz_index = None
        self.alpha_2_nz_index = None

        # set target coefficient values
        self.target_coeff_1 = target_coeff_1
        self.target_coeff_2 = target_coeff_2
        self.target_coeff_3 = target_coeff_3

        self.reg_param = reg_param
        self.eim_rank = eim_rank
        self.npts = npts
        self.seed = seed
        self.learn_rate = learn_rate

        self.n_ugrid = n_ugrid
        self.x_ugrid = np.linspace(0.0, 2.0*np.pi, n_ugrid, dtype=np.float32)
        self.t_ugrid = np.linspace(0.0, 1.0, n_ugrid, dtype=np.float32)

        self.data_target_coeff_1 = 20
        ###################### Dataset #######################
        if pde_type == "convection":
            self.train_data_f    = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_f_15_{pde_type}.csv')
        elif pde_type == "cdr":
            self.train_data_f    = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_f_1_1_1_{pde_type}.csv')
        #self.train_data_u    = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_u_{target_coeff_1}_{pde_type}.csv')
        #self.train_data_bd   = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_boundary_{target_coeff_1}_{pde_type}.csv')
        #self.test_data       = pd.read_csv(f'./data_gen/dataset/{pde_type}/test/test_{target_coeff_1}_{pde_type}.csv')
        ######################################################


        #mse_cost_function = torch.nn.MSELoss() # Mean squared error

        ############### Network Initialization ################
        net_initial = LR_PINN_phase1(hidden_dim, alpha_dim=alpha_dim)

        net_initial.load_state_dict(torch.load(f'./param/phase1/{pde_type}/{initial_condition}/PINN_{start_coeff_1}_{end_coeff_1}_{load_epoch}_h{hidden_dim}_' + "reg{:1.4e}".format(reg_param).replace(".","o") +  '.pt', map_location=device))

        self.start_w = net_initial.state_dict()['start_layer.weight']
        self.start_b = net_initial.state_dict()['start_layer.bias']
        self.end_w = net_initial.state_dict()['end_layer.weight']
        self.end_b = net_initial.state_dict()['end_layer.bias']

        self.col_0 = net_initial.state_dict()['col_basis_0']
        self.col_1 = net_initial.state_dict()['col_basis_1']
        self.col_2 = net_initial.state_dict()['col_basis_2']

        self.row_0 = net_initial.state_dict()['row_basis_0']
        self.row_1 = net_initial.state_dict()['row_basis_1']
        self.row_2 = net_initial.state_dict()['row_basis_2']

        self.meta_layer_1_w = net_initial.state_dict()['meta_layer_1.weight']
        self.meta_layer_1_b = net_initial.state_dict()['meta_layer_1.bias']
        self.meta_layer_2_w = net_initial.state_dict()['meta_layer_2.weight']
        self.meta_layer_2_b = net_initial.state_dict()['meta_layer_2.bias']
        self.meta_layer_3_w = net_initial.state_dict()['meta_layer_3.weight']
        self.meta_layer_3_b = net_initial.state_dict()['meta_layer_3.bias']    

        self.meta_alpha_0_w = net_initial.state_dict()['meta_alpha_0.weight']
        self.meta_alpha_0_b = net_initial.state_dict()['meta_alpha_0.bias']
        self.meta_alpha_1_w = net_initial.state_dict()['meta_alpha_1.weight']
        self.meta_alpha_1_b = net_initial.state_dict()['meta_alpha_1.bias']
        self.meta_alpha_2_w = net_initial.state_dict()['meta_alpha_2.weight']
        self.meta_alpha_2_b = net_initial.state_dict()['meta_alpha_2.bias']

        # train_data_f is needed to read off collocation points
        if pde_type == "convection":
            train_data_fpath = os.path.join(
                                    "data_gen", 
                                    "dataset",
                                    f"{pde_type}",
                                    "train",
                                    f"train_f_{target_coeff_1}_{pde_type}.csv"
                                    )
        elif pde_type == "cdr":
            train_data_fpath = os.path.join(
                                    "data_gen", 
                                    "dataset",
                                    f"{pde_type}",
                                    "train",
                                    f"train_f_1_1_1_{pde_type}.csv"
                                    )
        self.train_data_f    = pd.read_csv(train_data_fpath)
        #self.train_data_u    = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_u_{target_coeff_1}_{pde_type}.csv')
        #self.train_data_bd   = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_boundary_{target_coeff_1}_{pde_type}.csv')
        #self.test_data       = pd.read_csv(f'./data_gen/dataset/{pde_type}/test/test_{target_coeff_1}_{pde_type}.csv')

        self.x_collocation = torch.from_numpy(np.array(np.expand_dims(self.train_data_f['x_data'], 1))).float()
        self.t_collocation = torch.from_numpy(np.array(np.expand_dims(self.train_data_f['t_data'], 1))).float()
        self.beta_collocation = torch.from_numpy(np.array(np.expand_dims(self.train_data_f['beta'], 1))).float()
        self.nu_collocation = torch.from_numpy(np.array(np.expand_dims(self.train_data_f['nu'], 1))).float()
        self.rho_collocation = torch.from_numpy(np.array(np.expand_dims(self.train_data_f['rho'], 1))).float()

    
    def eim_row_index(self, col):

        r = col.shape[1]

        if type(col) == torch.Tensor:
            newcol = col.clone()
            row_index = torch.zeros(r, dtype=torch.long)
            argmax = torch.argmax
            abs = torch.abs
        elif type(col) == np.ndarray:
            newcol = col.copy()
            row_index = np.zeros(r, dtype=int)
            argmax = np.argmax
            abs = np.abs
        else:
            raise TypeError("unrecognized type")

        for i in range(r):
            v = newcol[:, i]

            i0 = argmax(abs(v)).item()
            row_index[i] = i0

            for j in range(i+1, r):
                w = newcol[:, j] 
                newcol[:, j] = w - w[i0]/v[i0] * v

        return row_index


    def meta_NN(self, target_coeff):
        """
        meta network
        """

        tanh = nn.Tanh()
        relu = nn.ReLU()

        meta_layer_1_w = self.meta_layer_1_w
        meta_layer_2_w = self.meta_layer_2_w
        meta_layer_3_w = self.meta_layer_3_w

        meta_layer_1_b = self.meta_layer_1_b
        meta_layer_2_b = self.meta_layer_2_b
        meta_layer_3_b = self.meta_layer_3_b

        meta_alpha_0_w = self.meta_alpha_0_w
        meta_alpha_1_w = self.meta_alpha_1_w
        meta_alpha_2_w = self.meta_alpha_2_w

        meta_alpha_0_b = self.meta_alpha_0_b
        meta_alpha_1_b = self.meta_alpha_1_b
        meta_alpha_2_b = self.meta_alpha_2_b

        meta_vector = torch.matmul(target_coeff, meta_layer_1_w.T) + meta_layer_1_b
        meta_vector = tanh(meta_vector)

        meta_vector = torch.matmul(meta_vector, meta_layer_2_w.T) + meta_layer_2_b
        meta_vector = tanh(meta_vector)

        meta_vector = torch.matmul(meta_vector, meta_layer_3_w.T) + meta_layer_3_b
        meta_vector = tanh(meta_vector)

        alpha_0 = relu(torch.matmul(meta_vector, meta_alpha_0_w.T) + meta_alpha_0_b)
        alpha_1 = relu(torch.matmul(meta_vector, meta_alpha_1_w.T) + meta_alpha_1_b)
        alpha_2 = relu(torch.matmul(meta_vector, meta_alpha_2_w.T) + meta_alpha_2_b)

        return alpha_0, alpha_1, alpha_2

    def get_truncated_net(self, coeff0):

        target_coeff = torch.tensor([coeff0, 0.0, 0.0])
        alpha_0, alpha_1, alpha_2 = self.metaNN(target_coeff)

        alpha_0_nz_index = self.alpha_0_nz_index
        alpha_1_nz_index = self.alpha_1_nz_index
        alpha_2_nz_index = self.alpha_2_nz_index

        alpha_0 = torch.gather(input=alpha_0, dim=0, index=alpha_0_nz_index)
        alpha_1 = torch.gather(input=alpha_1, dim=0, index=alpha_1_nz_index)
        alpha_2 = torch.gather(input=alpha_2, dim=0, index=alpha_2_nz_index) 

        # draft...

    def compute_truncated_alpha_range(self, coeff_tensor):

        alpha_0, alpha_1, alpha_2 = self.meta_NN(coeff_tensor)

        alpha_0 = alpha_0.detach().numpy()
        alpha_1 = alpha_1.detach().numpy()
        alpha_2 = alpha_2.detach().numpy()

        alpha_0_ii = (alpha_0 > 0).any(axis=0)
        alpha_1_ii = (alpha_1 > 0).any(axis=0)
        alpha_2_ii = (alpha_2 > 0).any(axis=0)

        self.alpha_0_nz_index = torch.arange(100)[alpha_0_ii]
        self.alpha_1_nz_index = torch.arange(100)[alpha_1_ii]
        self.alpha_2_nz_index = torch.arange(100)[alpha_2_ii]

        print(self.alpha_0_nz_index)
        print(self.alpha_1_nz_index)
        print(self.alpha_2_nz_index)


    def set_uniform_eim_pts(self, nx=4, nt=3):
        """

        Pre-set eim points, uniformly scattered
        
        """

        x_tensor = torch.linspace(0.0, 2.0*torch.pi, nx)
        t_tensor = torch.linspace(0.0, 1.0, nt)
        x_pts, t_pts = torch.meshgrid(x_tensor, t_tensor, indexing="ij")

        index_init = torch.zeros_like(x_pts, dtype=bool)
        index_init[:, 0] = True
        index_lb = torch.zeros_like(x_pts, dtype=bool)
        index_lb[0, :] = True
        index_ub = torch.zeros_like(x_pts, dtype=bool)
        index_ub[-1, :] = True

        index_interior = ~(index_init + index_lb + index_ub)

        self.index_init = index_init.flatten()
        self.index_lb = index_lb.flatten()
        self.index_ub = index_ub.flatten()
        self.index_interior = index_interior.flatten()

        x_pts = x_pts.reshape(-1, 1)
        t_pts = t_pts.reshape(-1, 1)

        self.x_collocation_fast = x_pts
        self.t_collocation_fast = t_pts


    def compute_truncated_alpha(self, coeff_tensor):
        """
        compute indices of non-zero entries in meta NN output alpha,
        for given sample values of coeff_tensor

        Parameters
        ----------
        coeff_tensor : tensor-like, (?, 3)

        """

        alpha_0, alpha_1, alpha_2 = self.meta_NN(coeff_tensor)

        alpha_0 = alpha_0.detach().numpy()
        alpha_1 = alpha_1.detach().numpy()
        alpha_2 = alpha_2.detach().numpy()

        alpha_0_ii = (alpha_0 > 0).any(axis=0)
        alpha_1_ii = (alpha_1 > 0).any(axis=0)
        alpha_2_ii = (alpha_2 > 0).any(axis=0)

        self.alpha_0_nz_index = torch.arange(len(alpha_0_ii))[alpha_0_ii]
        self.alpha_1_nz_index = torch.arange(len(alpha_1_ii))[alpha_1_ii]
        self.alpha_2_nz_index = torch.arange(len(alpha_2_ii))[alpha_2_ii]


    def compute_truncated_meta_alpha(self, coeff_tensor):
        """
        run after compute_truncated_alpha_range

        get meta NN output alphas, but in truncated (short) tensor format,
        also select corresponding columns in row and col basis

        Parameters
        ----------
        coeff_tensor : tensor-like, (?, 3)

        The truncated alphas, rows, cols, are stored in self._alpha_?,
        self._col_?, self._row_? respectively. 

        """

        if (self.alpha_0_nz_index is None) or \
           (self.alpha_1_nz_index is None) or \
           (self.alpha_2_nz_index is None): 
           raise ValueError("Don't know how to truncate: run compute_truncated_alpha() first")

        alpha_0_nz_index = self.alpha_0_nz_index
        alpha_1_nz_index = self.alpha_1_nz_index
        alpha_2_nz_index = self.alpha_2_nz_index

        alpha_0, alpha_1, alpha_2 = self.meta_NN(coeff_tensor)

        self._alpha_0 = torch.gather(input=alpha_0, dim=0,
                               index=alpha_0_nz_index)
        self._alpha_1 = torch.gather(input=alpha_1, dim=0,
                               index=alpha_1_nz_index)
        self._alpha_2 = torch.gather(input=alpha_2, dim=0,
                               index=alpha_2_nz_index)

        self._col_0 = torch.index_select(
                                    input=self.col_0,
                                    dim=1, 
                                    index=alpha_0_nz_index,
                                    )
        self._col_1 = torch.index_select(
                                    input=self.col_1,
                                    dim=1, 
                                    index=alpha_1_nz_index,
                                    )
        self._col_2 = torch.index_select(
                                    input=self.col_2,
                                    dim=1, 
                                    index=alpha_2_nz_index,
                                    )

        self._row_0 = torch.index_select(
                                    input=self.row_0,
                                    dim=1, 
                                    index=alpha_0_nz_index,
                                    )
        self._row_1 = torch.index_select(
                                    input=self.row_1,
                                    dim=1,
                                    index=alpha_1_nz_index,
                                    )
        self._row_2 = torch.index_select(
                                    input=self.row_2,
                                    dim=1, 
                                    index=alpha_2_nz_index,
                                    )

    def load_eim(self):
        row_index = np.loadtxt("eim_index.txt")
        eim_basis = np.load("eim_basis.npy")

        self.eim_index = torch.tensor(row_index, dtype=torch.long)
        self.eim_basis = torch.tensor(eim_basis, dtype=torch.float32)


    def load_phase_fast(self, coeff0):

        target_coeff = torch.tensor([coeff0, 0.0, 0.0])

        self.compute_truncated_meta_alpha(target_coeff)

        net1 = LR_PINN_phase_fast(
                                  self.hidden_dim,
                                  self.start_w,
                                  self.start_b,
                                  self.end_w,
                                  self.end_b,
                                  self._col_0,
                                  self._col_1,
                                  self._col_2,
                                  self._row_0,
                                  self._row_1,
                                  self._row_2,
                                  self._alpha_0,
                                  self._alpha_1,
                                  self._alpha_2,
                                  self.eim_index, 
                                  self.eim_basis,
                                  self.eim_rank,
                                  )
        net1.to(self.device)
        self.net_fast = net1


    def load_phase_fast_target(self, coeff_tensor):
        """
        Parameters
        ----------
        coeff_tensor : tensor-like, shape (1, 3)
        """

        self.compute_truncated_meta_alpha(coeff_tensor)

        net1 = LR_PINN_phase_fast(
                                  self.hidden_dim,
                                  self.start_w,
                                  self.start_b,
                                  self.end_w,
                                  self.end_b,
                                  self._col_0,
                                  self._col_1,
                                  self._col_2,
                                  self._row_0,
                                  self._row_1,
                                  self._row_2,
                                  self._alpha_0,
                                  self._alpha_1,
                                  self._alpha_2,
                                  self.eim_index, 
                                  self.eim_basis,
                                  self.eim_rank,
                                  )
        net1.to(self.device)
        self.net_fast = net1


    def load_phase2_target(self, target_coeff):

        self.compute_truncated_meta_alpha(target_coeff)

        net0 = LR_PINN_phase2(
                              self.hidden_dim,
                              self.start_w,
                              self.start_b,
                              self.end_w,
                              self.end_b,
                              self._col_0,
                              self._col_1,
                              self._col_2,
                              self._row_0,
                              self._row_1,
                              self._row_2, 
                              self._alpha_0, 
                              self._alpha_1, 
                              self._alpha_2,
                              )
        net0.to(self.device)
        net0.eval()
        self.net_phase2 = net0


    def load_phase2(self, coeff_tensor):
        """
        load phase2 LRNR with given coefficients into self.net_phase2 

        """

        self.compute_truncated_meta_alpha(coeff_tensor)

        net0 = LR_PINN_phase2(
                              self.hidden_dim,
                              self.start_w,
                              self.start_b,
                              self.end_w,
                              self.end_b,
                              self._col_0,
                              self._col_1,
                              self._col_2,
                              self._row_0,
                              self._row_1,
                              self._row_2, 
                              self._alpha_0, 
                              self._alpha_1, 
                              self._alpha_2,
                              )
        
        net0.to(self.device)
        net0.eval()
        self.net_phase2 = net0

    def coeff1_collocation(self, coeff_tensor, device="cpu"):

        x_collocation    = Variable(self.x_collocation, requires_grad=True).to(device)
        t_collocation    = Variable(self.t_collocation, requires_grad=True).to(device)
        beta_collocation = Variable(torch.ones_like(x_collocation)*coeff_tensor[0], requires_grad=True).to(device)
        nu_collocation   = Variable(torch.ones_like(x_collocation)*coeff_tensor[1], requires_grad=True).to(device)
        rho_collocation  = Variable(torch.ones_like(x_collocation)*coeff_tensor[2], requires_grad=True).to(device)

        return x_collocation, t_collocation, beta_collocation, nu_collocation, rho_collocation


    def load_phase2_(self):

        net0 = LR_PINN_phase2(
                              self.hidden_dim,
                              self.start_w,
                              self.start_b,
                              self.end_w,
                              self.end_b,
                              self._col_0,
                              self._col_1,
                              self._col_2,
                              self._row_0,
                              self._row_1,
                              self._row_2, 
                              self._alpha_0, 
                              self._alpha_1, 
                              self._alpha_2,
                              )
        net0.to(self.device)
        net0.eval()
        self.net_phase2 = net0

    def compute_eim(self,
                    coeff_tensor,
                    n=50,
                    npts=5,
                    delta=1e-4,
                    ):

        if os.path.exists("eim_basis.npy") \
            and os.path.exists("eim_singv.npy") \
                and os.path.exists("eim_index.npy") \
                    and os.path.exists("eim_snapshot_error.npy"):
             warnings.warn("Found existing basis")
             self.load_eim()
             return True

        self.npts = npts

        x = self.x_collocation_fast[:npts, :].detach().clone()
        t = self.t_collocation_fast[:npts, :].detach().clone()

        input0 = torch.cat([x, t], axis=1)
        input1 = torch.cat([x + delta, t], axis=1)
        input2 = torch.cat([x , t + delta], axis=1)
        input3 = torch.cat([x - delta, t], axis=1)
        input4 = torch.cat([x , t - delta], axis=1)

        inputs = torch.cat([input0, input1, input2, input3, input4], axis=0)
        hidden_dim = self.hidden_dim
        tanh = nn.Tanh()

        snapshots_0_list = []
        snapshots_1_list = []
        snapshots_2_list = []
        snapshots_3_list = []

        self.compute_truncated_alpha(coeff_tensor)

        n = coeff_tensor.shape[0]

        for target_coeff in coeff_tensor:

            self.compute_truncated_meta_alpha(target_coeff)

            col_0 = self._col_0
            col_1 = self._col_1
            col_2 = self._col_2

            alpha_0 = self._alpha_0
            alpha_1 = self._alpha_1
            alpha_2 = self._alpha_2

            ## phase2
            coeff_0 = torch.matmul(col_0, torch.diag(alpha_0))
            coeff_1 = torch.matmul(col_1, torch.diag(alpha_1))
            coeff_2 = torch.matmul(col_2, torch.diag(alpha_2))

            start_layer = nn.Linear(2, hidden_dim)
            end_layer = nn.Linear(hidden_dim, 1)
            
            start_layer.weight = nn.Parameter(self.start_w)
            start_layer.bias = nn.Parameter(self.start_b)

            end_layer.weight = nn.Parameter(self.end_w)
            end_layer.bias = nn.Parameter(self.end_b)

            emb_out = start_layer(inputs)
            emb_out = tanh(emb_out)

            self.emb_out_0 = emb_out.detach().clone()
            snapshots_0_list.append(emb_out.detach().numpy())        ##

            emb_out = torch.matmul(emb_out, coeff_0)
            self._phase2_coeff_0 = emb_out.detach().clone()

            emb_out = torch.matmul(emb_out, self._row_0.T)
            emb_out = tanh(emb_out)

            self.emb_out_1 = emb_out.detach().clone()
            snapshots_1_list.append(emb_out.detach().numpy())       ##

            emb_out = torch.matmul(emb_out, coeff_1)
            self._phase2_coeff_1 = emb_out.detach().clone()

            emb_out = torch.matmul(emb_out, self._row_1.T)
            emb_out = tanh(emb_out)

            self.emb_out_2 = emb_out.detach().clone()
            snapshots_2_list.append(emb_out.detach().numpy())       ##

            emb_out = torch.matmul(emb_out, coeff_2)

            self._phase2_coeff_2 = emb_out.detach().clone()

            emb_out = torch.matmul(emb_out, self._row_2.T)
            emb_out = tanh(emb_out)

            self._phase2_emb_out_3 = emb_out.detach().clone()
            snapshots_3_list.append(emb_out.detach().numpy())       ##

            emb_out = end_layer(emb_out)

        self.snapshots = [
                            np.vstack(snapshots_0_list).T,
                            np.vstack(snapshots_1_list).T,
                            np.vstack(snapshots_2_list).T,
                            np.vstack(snapshots_3_list).T,
                        ]

        self.eim_basis = torch.zeros((4, hidden_dim, n))
        self.eim_index = torch.zeros((4, n), dtype=int)
        self.eim_singv = torch.zeros((4, n))
        self.snapshot_error = np.zeros(4)

        for i in range(4):
            u0, s0, v0 = np.linalg.svd(self.snapshots[i], full_matrices=False)
            self.eim_basis[i, ...] = torch.from_numpy(u0[:, :n])
            self.eim_singv[i, ...] = torch.from_numpy(s0[:n])
            self.eim_index[i, :] = self.eim_row_index(self.eim_basis[i, ...])
            self.snapshot_error[i] = \
                    self.eim_snapshot_error(
                            self.eim_basis[i, ...],
                            self.eim_index[i, ...],
                            self.snapshots[i],
                            )

        # save eim info
        np.save("eim_basis.npy", self.eim_basis.detach().numpy())
        np.save("eim_singv.npy", self.eim_singv.detach().numpy())
        np.save("eim_index.npy", self.eim_index.detach().numpy())
        np.save("eim_snapshot_error.npy", self.snapshot_error)


    def load_eim(self):
        self.eim_basis = torch.from_numpy(np.load("eim_basis.npy"))
        self.eim_singv = torch.from_numpy(np.load("eim_singv.npy"))
        self.eim_index = torch.from_numpy(np.load("eim_index.npy"))
        self.snapshot_error = np.load("eim_snapshot_error.npy")


    def eim_snapshot_error(self, U0, i0, S0):
        c0 = np.linalg.solve(U0[i0, :], S0[i0, :])
        return np.linalg.norm(U0 @ c0 - S0, ord=np.inf)


    def eval_phase2_grid(self):

        X, T = np.meshgrid(self.x_ugrid, self.t_ugrid)

        old_shape = X.shape

        X = torch.from_numpy(X.reshape(-1, 1))
        T = torch.from_numpy(T.reshape(-1, 1))

        self.X = X
        self.T = T
        self.net_phase2.eval()
        out = self.net_phase2(X, T)
        out = out.reshape(old_shape)
        return out

    def eval_ugrid_npy(self, net, eval=False):

        X, T = np.meshgrid(self.x_ugrid, self.t_ugrid)

        old_shape = X.shape

        self.X_meshgrid = X
        self.T_meshgrid = T

        X = torch.from_numpy(X.reshape(-1, 1))
        T = torch.from_numpy(T.reshape(-1, 1))

        if eval: net.eval()

        out = net(X, T)
        out = out.reshape(old_shape)
        out = out.detach().numpy()
        return out

    def eval_ugrid_tensor(self, net, eval=False):

        X, T = np.meshgrid(self.x_ugrid, self.t_ugrid)

        old_shape = X.shape

        X = torch.from_numpy(X.reshape(-1, 1))
        T = torch.from_numpy(T.reshape(-1, 1))

        self.X_meshgrid_tensor = X
        self.T_meshgrid_tensor = T
        X.requires_grad = True
        T.requires_grad = True

        if eval: net.eval()

        out = net(X, T)
        #out = out.reshape(old_shape)
        return out


    def eval_colloc_tensor(self, net, eval=False):

        ii = self.index_interior
        x_colloc = self.x_collocation_fast.clone()[ii, :]
        t_colloc = self.t_collocation_fast.clone()[ii, :]

        x_colloc.requires_grad = True
        t_colloc.requires_grad = True

        self.x_colloc = x_colloc
        self.t_colloc = t_colloc

        if eval: net.eval()

        out = net(x_colloc, t_colloc)
        #out = out.reshape(old_shape)
        return out



    def eim_check(self, coeff0,
                  x=torch.tensor([[np.pi]], dtype=torch.float32),
                  t=torch.tensor([[0.5]], dtype=torch.float32),
                ):
        """
        check difference between eim and output 
        """

        target_coeff = torch.tensor([coeff0, 0.0, 0.0], dtype=torch.float32)
        self.compute_truncated_meta_alpha(target_coeff)

        inputs = torch.cat([x, t], axis=1)
        hidden_dim = self.hidden_dim
        tanh = nn.Tanh()

        ## proper phase2

        coeff_0 = torch.matmul(self._col_0, torch.diag(self._alpha_0))
        coeff_1 = torch.matmul(self._col_1, torch.diag(self._alpha_1))
        coeff_2 = torch.matmul(self._col_2, torch.diag(self._alpha_2))

        start_layer = nn.Linear(2, hidden_dim)
        end_layer = nn.Linear(hidden_dim, 1)
        
        start_layer.weight = nn.Parameter(self.start_w)
        start_layer.bias = nn.Parameter(self.start_b)

        end_layer.weight = nn.Parameter(self.end_w)
        end_layer.bias = nn.Parameter(self.end_b)

        emb_out = start_layer(inputs)
        emb_out = tanh(emb_out)

        self.emb_out_0 = emb_out.detach().clone()

        _, eim_0 = self.eim_coeff(self.emb_out_0, k=0, return_proj=True)
        error_0 = np.linalg.norm(self.emb_out_0.detach().numpy() \
                                 - eim_0.detach().numpy(), ord=np.inf)
        print("error 0 = {:1.6f}".format(error_0))

        emb_out = torch.matmul(emb_out, coeff_0)
        self._phase2_coeff_0 = emb_out.detach().clone()

        emb_out = torch.matmul(emb_out, self._row_0.T)
        emb_out = tanh(emb_out)

        self.emb_out_1 = emb_out.detach().clone()

        _, eim_1 = self.eim_coeff(self.emb_out_1, k=1, return_proj=True)

        error_1 = np.linalg.norm(self.emb_out_1.detach().numpy() \
                                 - eim_1.detach().numpy(), ord=np.inf)
        print("error 1 = {:1.6f}".format(error_1))

        emb_out = torch.matmul(emb_out, coeff_1)
        self._phase2_coeff_1 = emb_out.detach().clone()

        emb_out = torch.matmul(emb_out, self._row_1.T)
        emb_out = tanh(emb_out)

        self.emb_out_2 = emb_out.detach().clone()
        _, eim_2 = self.eim_coeff(self.emb_out_2, k=2, return_proj=True)
        error_2 = np.linalg.norm(self.emb_out_2.detach().numpy() \
                                 - eim_2.detach().numpy(), ord=np.inf)
        print("error 2 = {:1.6f}".format(error_2))

        emb_out = torch.matmul(emb_out, coeff_2)

        self._phase2_coeff_2 = emb_out.detach().clone()

        emb_out = torch.matmul(emb_out, self._row_2.T)
        emb_out = tanh(emb_out)

        self.emb_out_3 = emb_out.detach().clone()

        _, eim_3 = self.eim_coeff(self.emb_out_3, k=3, return_proj=True)
        error_3 = np.linalg.norm(self.emb_out_3.detach().numpy() - eim_3.detach().numpy(), ord=np.inf)
        print("error 3 = {:1.6f}".format(error_3))

        self._phase2_emb_out_3 = emb_out.detach().clone()

        emb_out = end_layer(emb_out)


    def eim_coeff(self, w, k=0, return_proj=False):

        eim_basis = self.eim_basis
        eim_index = self.eim_index
        eim_rank = self.eim_rank

        ii = eim_index[k, :eim_rank]
        eim_mat = eim_basis[k, ii, :eim_rank]

        coeff = np.linalg.solve(eim_mat, w[:, ii].T)

        if return_proj:
            proj = (eim_basis[k, :, :eim_rank]  @ coeff).T
            return coeff, proj
        else:
            return coeff


    def eval_phase2_fast_grid(self):

        X, T = np.meshgrid(self.x_ugrid, self.t_ugrid)

        X = torch.from_numpy(X.reshape(-1, 1))
        T = torch.from_numpy(T.reshape(-1, 1))

        self.X = X
        self.T = T

        self.net_phase2.eval()
        self.net_fast.eval()

        self._phase2_out = self.net_phase2(X, T)
        self._fast_out = self.net_fast(X, T)


    def plot_phase2_fast(self):

        self.eval_phase2_fast_grid()

        X = self.X.detach().numpy().reshape(128,128)
        T = self.T.detach().numpy().reshape(128,128)
        phase2_out = self._phase2_out.detach().numpy().reshape(128,128)
        fast_out = self._fast_out.detach().numpy().reshape(128,128)

        fig, axs = plt.subplots(ncols=3, figsize=(12, 4))
        ax = axs[0]
        im = ax.pcolormesh(X, T, phase2_out)
        plt.colorbar(im, ax=ax)
        ax = axs[1]
        im = ax.pcolormesh(X, T, fast_out)
        plt.colorbar(im, ax=ax)
        ax = axs[2]
        im = ax.pcolormesh(X, T, phase2_out - fast_out)
        plt.colorbar(im, ax=ax)

        return fig


    def compare_fast_phase2(self, coeff=18.1, dx=1e-4):

        self.load_phase_fast(coeff)
        self.load_phase2(coeff)

        npts = self.npts

        x0 = self.x_collocation[:npts, :] 
        t0 = self.t_collocation[:npts, :] 
        x1 = self.x_collocation[:npts, :] + dx
        t1 = self.t_collocation[:npts, :]  
        x2 = self.x_collocation[:npts, :]  
        t2 = self.t_collocation[:npts, :] + dx

        x = torch.cat([x0, x1, x2], dim=0)
        t = torch.cat([t0, t1, t2], dim=0)

        self.net_fast.eval()
        self.net_phase2.eval()

        out_fast = self.net_fast(x, t)
        out_phase2 = self.net_phase2(x, t)
        
        print(out_fast - out_phase2)

        self.net_fast.alpha_0.data += dx
        self.net_fast.alpha_1.data += dx
        self.net_fast.alpha_2.data += dx

        self.net_phase2.alpha_0.data += dx
        self.net_phase2.alpha_1.data += dx
        self.net_phase2.alpha_2.data += dx

        out_fast_dalph = self.net_fast(x, t)
        out_phase2_dalph = self.net_phase2(x, t)

        print(out_fast_dalph - out_phase2_dalph)

        print(out_fast - out_fast_dalph)
        print(out_phase2 - out_phase2_dalph)

    def _tonumpy(self, a):
        return a.detach().numpy()


    def train_fast(self,
                   coeff_tensor,
                   epoch=200,
                   test_device = "cuda",
                   reg_param=1.0,
                   device="cpu",
                   ):

        get_target_str = self.get_target_str
        compute_numerical_solution = self.compute_numerical_solution
        set_uniform_eim_pts = self.set_uniform_eim_pts
        load_phase_fast_target = self.load_phase_fast_target

        target_str = get_target_str(coeff_tensor)

        npts = self.npts
        pde_type = self.pde_type

        tonumpy = self._tonumpy

        err_list = [] 
        ep_list = []
        loss_list= []
        mse_loss_list = []

        mse_u_list = []
        mse_f_list = []
        mse_bd_list = []

        L1_rel_err_list = []
        L2_rel_err_list = []
        Max_err_list = []
        Ex_var_score_list = []

        time_list = []

        x_col0    = self.x_collocation_fast
        t_col0    = self.t_collocation_fast

        #test_data = pd.read_csv(f'./data_gen/dataset/{pde_type}/test/test_{target_coeff_1}_{pde_type}.csv')
        
        # test point 
        L = 2*np.pi
        T = 1
        nx = 256
        nt = 100
        dx = L/nx
        dt = T/nt
        x = np.arange(0, L, dx) # not inclusive of the last point
        t = np.linspace(0, T, nt).reshape(-1, 1)
        X, T = np.meshgrid(x, t)
        x_test = torch.from_numpy(X.reshape(-1, 1)).float().to(test_device)
        t_test = torch.from_numpy(T.reshape(-1, 1)).float().to(test_device)

        self.x_test = x_test
        self.t_test = t_test

        u_test = compute_numerical_solution(coeff_tensor,
                                            pde_type,
                                            device=test_device)

        L1_true_norm = torch.linalg.norm(u_test, 1, dim = 0)
        L2_true_norm = torch.linalg.norm(u_test, 2, dim = 0)

        self.load_phase_fast_target(coeff_tensor)
        net = self.net_fast.to(device)
        
        alpha_0_metaout = net.alpha_0.detach().clone()
        alpha_1_metaout = net.alpha_1.detach().clone()
        alpha_2_metaout = net.alpha_2.detach().clone()

        lr0 = self.learn_rate / 2

        # set up collocation / init / bd pts
        i0 = self.index_init
        ilb = self.index_lb
        iub = self.index_ub
        ii = self.index_interior

        x_col = Variable(x_col0[ii, :], requires_grad=True).to(device)
        t_col = Variable(t_col0[ii, :], requires_grad=True).to(device)
        beta_col = Variable(torch.ones_like(x_col)*coeff_tensor[0],
                                requires_grad=True).to(device)
        nu_col = Variable(torch.ones_like(x_col)*coeff_tensor[1],
                                requires_grad=True).to(device)
        rho_col = Variable(torch.ones_like(x_col)*coeff_tensor[2],
                                requires_grad=True).to(device)

        x_lb = Variable(x_col0[ilb, :], requires_grad=True).to(device)
        t_lb = Variable(t_col0[ilb, :], requires_grad=True).to(device)

        x_ub = Variable(x_col0[iub, :], requires_grad=True).to(device)
        t_ub = Variable(t_col0[iub, :], requires_grad=True).to(device)

        x_init = Variable(x_col0[i0, :], requires_grad=True).to(device)
        t_init = Variable(t_col0[i0, :], requires_grad=True).to(device)
        u0 = systems.function("1+sin(x)")
        u_init = torch.from_numpy(u0(x_init.detach().numpy())).to(device)

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=lr0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.95)

        mse_cost_function = torch.nn.MSELoss() # Mean squared error

        for ep in range(0, epoch+1):

            net.train()

            t0 = time.time()
            optimizer.zero_grad()

            net_initial_out = net(x_init, t_init)
            mse_u = torch.sum(torch.abs(net_initial_out - u_init))
            
            f_out = f_cal_phase2(x_col, t_col, beta_col, nu_col, rho_col, net)
            mse_f = torch.sum(torch.abs(f_out))

            u_pred_lb = net(x_lb, t_lb)
            u_pred_ub = net(x_ub, t_ub)
            
            #mse_bd = torch.sum(torch.abs(u_pred_lb - u_pred_ub))

            #loss = torch.max(torch.cat([mse_f.reshape(1,1),
            #                            mse_u.reshape(1,1), 
            #                            mse_bd.reshape(1,1)]))

            #intv = 4
            #if (ep // intv) % 3 == 0:
            #    loss = mse_f
            #elif (ep // intv) % 3 == 1:
            #    loss = mse_f
            #elif (ep // intv) % 3 == 2:
            #    loss = mse_u

            #loss = mse_f + mse_u + mse_bd
            loss = mse_f + mse_u 

            # alpha-localization 
            reg = reg_param*\
                  (torch.sum(torch.abs(net.alpha_0 - alpha_0_metaout)) \
                 + torch.sum(torch.abs(net.alpha_1 - alpha_1_metaout)) \
                 + torch.sum(torch.abs(net.alpha_2 - alpha_2_metaout)) )

            loss += reg

            loss.backward()

            if ep > 1:
                scheduler.step(loss)
                optimizer.step()

            # alpha_0 = tonumpy(net.alpha_0)
            # alpha_1 = tonumpy(net.alpha_1)
            # alpha_2 = tonumpy(net.alpha_2)

            # alpha_norm = np.linalg.norm(alpha_0)**2 \
            #            + np.linalg.norm(alpha_1)**2 \
            #            + np.linalg.norm(alpha_2)**2

            t1 = time.time()

            for g in optimizer.param_groups:
                lr0 = g["lr"]
            
            if ep % 1 == 0:
                time_list.append(t1 - t0)
                net.eval()

                self._alpha_0 = net.alpha_0.detach().clone()
                self._alpha_1 = net.alpha_1.detach().clone()
                self._alpha_2 = net.alpha_2.detach().clone()

                self.load_phase2_()
                net0 = self.net_phase2
                net0.to(test_device)

                with torch.autograd.no_grad():

                    u_out_test = net0(x_test, t_test)

                    mse_test = mse_cost_function(u_out_test, u_test)
                    
                    err_list.append(mse_test.item())
                    ep_list.append(ep)
                    loss_list.append(loss.item())

                    mse_f_list.append(mse_f.item())

                    L1_error = torch.linalg.norm(u_out_test-u_test, 1, dim = 0)
                    L2_error = torch.linalg.norm(u_out_test-u_test, 2, dim = 0)
                    
                    L1_rel_err = L1_error / L1_true_norm
                    L2_rel_err = L2_error / L2_true_norm

                    u_test_cpu = u_test.cpu()
                    u_out_test_cpu = u_out_test.cpu()

                    Max_err = max_error(u_test_cpu, u_out_test_cpu)
                    Ex_var_score = explained_variance_score(u_test_cpu, u_out_test_cpu)

                    L1_rel_err_list.append(L1_rel_err.item())
                    L2_rel_err_list.append(L2_rel_err.item())
                    Max_err_list.append(Max_err)
                    Ex_var_score_list.append(Ex_var_score)
                    
                    print('-'*30 + " fast training " + '-'*30)
                    #print(net.alpha_2[1].item())

                    print('target : {:s} lr : {:1.4e} L1_rel : {:1.4e} L2_rel :{:1.4e}'.format(target_str, lr0, L1_rel_err.item(), L2_rel_err.item()))
                    # print('Max_err : {:1.4e}'.format(Max_err))
                    # print('Ex_var_score : {:1.4e}'.format(Ex_var_score))
                    print('reg_param: {:1.2f} Epoch : {:04d} Error : {:1.4e} train_loss (total) : {:1.4e}'.format(reg_param, ep, mse_test.item(), loss.item()))
                    # print('mse_f : {:1.4e} mse_u : {:1.4e} mse_bd : {:1.4e}'.format(mse_f.item(), mse_u.item(), mse_bd.item()))
                    #print("alpha norm: {:1.4f} reg: {:1.4f}".format(alpha_norm, reg.item()))


            SAVE_PATH = f'./log/phase2/{pde_type}/{self.initial_condition}'
            SAVE_NAME = 'PINN_fast_{:s}.npy'.format(target_str)

            save_fpath = os.path.join(SAVE_PATH, SAVE_NAME)
            
            if not os.path.isdir(SAVE_PATH): os.mkdir(SAVE_PATH)
            np.save(save_fpath,
                    np.array([
                                L1_rel_err_list,
                                L2_rel_err_list,
                                loss_list,
                                time_list,
                                ]))

            SAVE_NAME = 'PINN_fast_sol_{:s}.npy'.format(target_str)
            save_fpath = os.path.join(SAVE_PATH, SAVE_NAME)
            np.save(save_fpath,
                    np.array([
                                x_test.cpu().detach().numpy().flatten(),
                                t_test.cpu().detach().numpy().flatten(),
                                u_out_test.cpu().detach().numpy().flatten(),
                                ]))




    def train_adap(self,
                   coeff_tensor,
                   epoch=200,
                   device = "cuda",
                   ):
        """
        
        target_tensor : tensor-like, shape=(?, 3)
        
        """
        

        get_target_str = self.get_target_str
        coeff1_collocation = self.coeff1_collocation
        compute_numerical_solution = self.compute_numerical_solution
        set_init_bd_pts = self.set_init_bd_pts
        pde_type = self.pde_type
        initial_condition = self.initial_condition

        SAVE_PATH = f'./log/phase2/{pde_type}/{initial_condition}'
        SAVE_NAME = "_".join(["PINN_adap", get_target_str(coeff_tensor)])

        error_fpath = os.path.join(SAVE_PATH, SAVE_NAME)

        # skip run if desired output file exists
        if os.path.exists(error_fpath):
            warnings.warn(
                "Requested output exists - skip \n file path {:s}".format(error_fpath))
            return True
        if not os.path.isdir(SAVE_PATH): os.mkdir(SAVE_PATH)
                
        # TODO: do we need to seed here?
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        cudnn.benchmark = False
        cudnn.deterministic = True
        
        #device = torch.device(args.device)
        #print("========================================")
        #print("Use Device :", device)
        #print("Available cuda devices :", torch.cuda.device_count())
        #print("Current cuda device :", torch.cuda.current_device())
        #print("Name of cuda device :", torch.cuda.get_device_name(device))
        #print("========================================")
        
        hidden_dim = self.hidden_dim
        alpha_dim = self.alpha_dim
        reg_param = self.reg_param
        load_epoch = 20000
        lr0 = self.learn_rate

        
        data_target_coeff_1 =  self.data_target_coeff_1
        ###################### Dataset #######################
        train_data_f    = self.train_data_f
        ######################################################
        
        mse_cost_function = torch.nn.MSELoss() # Mean squared error

        self.load_phase2(coeff_tensor)
        net = self.net_phase2.to(device)

        #optimizer = torch.optim.AdamW(net.parameters(), lr=lr0, weight_decay=1e-4)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr0, betas=(0.9, 0.99))
        #optimizer = torch.optim.SGD(net.parameters(), lr=lr0, momentum=0.99, weight_decay=1e-5)

        T_max = 100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max,
                                                               eta_min=0,
                                                               last_epoch=-1)

        x_collocation, t_collocation, beta_collocation, nu_collocation, rho_collocation = \
            coeff1_collocation(coeff_tensor, device=device)

        #test_data = pd.read_csv(f'./data_gen/dataset/{pde_type}/test/test_{target_coeff_1}_{pde_type}.csv')
        
        # test point 
        L = 2*np.pi
        T = 1
        nx = 256
        nt = 100
        dx = L/nx
        dt = T/nt
        x = np.arange(0, L, dx) # skip the last point
        t = np.linspace(0, T, nt).reshape(-1, 1)
        X, T = np.meshgrid(x, t)
        x_test = torch.from_numpy(X.reshape(-1, 1)).float().to(device)
        t_test = torch.from_numpy(T.reshape(-1, 1)).float().to(device)

        self.x_test = x_test
        self.t_test = t_test

        u_test = compute_numerical_solution(coeff_tensor, device=device)

        L1_true_norm = torch.linalg.norm(u_test, 1, dim = 0)
        L2_true_norm = torch.linalg.norm(u_test, 2, dim = 0)

        set_init_bd_pts(device=device)

        model_size = get_params(net)
        print(model_size)    

        all_zeros = np.zeros((len(train_data_f), 1))
        all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

        x_initial = self.x_initial
        t_initial = self.t_initial
        u_initial = self.u_initial

        x_lb = self.x_lb
        t_lb = self.t_lb
        x_ub = self.x_ub
        t_ub = self.t_ub

        err_list = []
        ep_list = []
        loss_list= []
        mse_loss_list = []

        mse_u_list = []
        mse_f_list = []
        mse_bd_list = []

        L1_rel_list = []
        L2_rel_list = []
        Max_err_list = []
        Ex_var_score_list = []
        time_list = []

        # start training
        for ep in range(epoch+1):

            net.train()

            t0 = time.time()
            optimizer.zero_grad()
            net_initial_out = net(x_initial, t_initial)
            mse_u = mse_cost_function(net_initial_out, u_initial)
            
            f_out = f_cal_phase2(x_collocation,
                                t_collocation,
                                beta_collocation,
                                nu_collocation,
                                rho_collocation,
                                net)
            
            mse_f = mse_cost_function(f_out, all_zeros)
            
            u_pred_lb = net(x_lb, t_lb)
            u_pred_ub = net(x_ub, t_ub)
            
            mse_bd = torch.mean((u_pred_lb - u_pred_ub) ** 2)
            
            loss = mse_u + mse_f + mse_bd 

            loss.backward()
            if ep > 1:
                optimizer.step()
                scheduler.step()

            t1 = time.time()

            if ep % 1 == 0:
                time_list.append(t1 - t0)
                net.eval()
                with torch.autograd.no_grad():
                    u_out_test = net(x_test, t_test)
                    mse_test = mse_cost_function(u_out_test, u_test)
                    
                    err_list.append(mse_test.item())
                    ep_list.append(ep)

                    #if ep > 0: loss_list.append(loss.item())
                    loss_list.append(loss.item())
                    #mse_loss_list.append((mse_u+mse_f+mse_bd).item())

                    #mse_u_list.append(mse_u.item())
                    #mse_f_list.append(mse_f.item())
                    #mse_bd_list.append(mse_bd.item())

                    L1_error = torch.linalg.norm(u_out_test-u_test, 1, dim = 0)
                    L2_error = torch.linalg.norm(u_out_test-u_test, 2, dim = 0)
                    
                    L1_relative_error = L1_error / L1_true_norm
                    L2_relative_error = L2_error / L2_true_norm

                    u_test_cpu = u_test.cpu()
                    u_out_test_cpu = u_out_test.cpu()
                    Max_err = max_error(u_test_cpu, u_out_test_cpu)
                    Ex_var_score = explained_variance_score(u_test_cpu, u_out_test_cpu)

                    L1_rel_list.append(L1_relative_error.item())
                    L2_rel_list.append(L2_relative_error.item())
                    Max_err_list.append(Max_err)
                    Ex_var_score_list.append(Ex_var_score)

                    print('-'*30 + " adap training " + '-'*30)
                    #print(net.alpha_0[1].item())
                    print('lr : {:1.4e}'.format(lr0))
                    print('L1_rel_err : {:1.4e}'.format(L1_relative_error.item()))
                    print('L2_rel_err :{:1.4e}'.format(L2_relative_error.item()))
                    print('Max_err : {:1.4e}'.format(Max_err))
                    print('Ex_var_score : {:1.4e}'.format(Ex_var_score))
                    if ep > 0 : print('Epoch : {:08d} Error : {:1.4e} train_loss (total) : {:1.4e}'.format(ep,mse_test.item(), loss.item()))

            error_array = np.array([
                                    L1_rel_list,
                                    L2_rel_list,
                                    loss_list,
                                    time_list,
                                    ])
            np.save(error_fpath, error_array)

            target_str = get_target_str(coeff_tensor)
            SAVE_NAME = 'PINN_adap_sol_{:s}.npy'.format(target_str)
            save_fpath = os.path.join(SAVE_PATH, SAVE_NAME)
            np.save(save_fpath,
                np.array([
                            x_test.cpu().detach().numpy().flatten(),
                            t_test.cpu().detach().numpy().flatten(),
                            u_out_test.cpu().detach().numpy().flatten(),
                            ]))


    def get_target_str(self, target_tensor):

        str_list = \
            ["t{:05.2f}".format(target_tensor[i].item()).replace(".", "-") \
             for i in range(len(target_tensor))]
        return "_".join(str_list)


    def compute_numerical_solution(self,
                               coeff_tensor,
                               pde_type=None,
                               device=None):

        beta, nu, rho = coeff_tensor

        if device is None:
            device=self.device

        # TODO: better way?
        beta = float(beta)
        nu = float(nu)
        rho = float(rho)

        if pde_type is None: pde_type = self.pde_type

        if pde_type == "convection":
            u_vals, _ = \
                systems.convection_diffusion_discrete_solution(
                    "1+sin(x)", 0.0, beta)
        elif pde_type == "cdr":
            u_vals, _ = \
                systems.convection_diffusion_reaction_discrete_solution(
                    "1+sin(x)", beta, nu, rho)

        u_test = torch.from_numpy(u_vals.reshape(-1, 1)).float().to(device)

        return u_test


    def set_fast_collocation_pts_uniform(self, nx=4, nt=3):

        x_tensor = torch.linspace(0.0, 2.0*torch.pi, nx)
        t_tensor = torch.linspace(0.0, 1.0, nt)
        x_pts, t_pts = torch.meshgrid(x_tensor, t_tensor, indexing="ij")

        index_init = torch.zeros_like(x_pts, dtype=bool)
        index_init[:, 0] = True
        index_lb = torch.zeros_like(x_pts, dtype=bool)
        index_lb[0, :] = True
        index_ub = torch.zeros_like(x_pts, dtype=bool)
        index_ub[-1, :] = True

        index_interior = ~(index_init + index_lb + index_ub)

        self.index_init = index_init.flatten()
        self.index_lb = index_lb.flatten()
        self.index_ub = index_ub.flatten()
        self.index_interior = index_interior.flatten()

        x_pts = x_pts.reshape(-1, 1)
        t_pts = t_pts.reshape(-1, 1)

        self.x_collocation_fast = x_pts
        self.t_collocation_fast = t_pts


    def set_init_bd_pts(self, device="cpu"):

        pde_type = self.pde_type
        data_target_coeff_1 = 20

        if self.pde_type == "convection":
            train_data_u    = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_u_{data_target_coeff_1}_{pde_type}.csv')
            train_data_bd   = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_boundary_{data_target_coeff_1}_{pde_type}.csv')
        elif self.pde_type == "cdr":
            train_data_u    = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_u_1_1_1_{pde_type}.csv')
            train_data_bd   = pd.read_csv(f'./data_gen/dataset/{pde_type}/train/train_boundary_1_1_1_{pde_type}.csv')

        # initial points
        self.x_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_u['x_data'], 1))).float(), requires_grad=True).to(device)
        self.t_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_u['t_data'], 1))).float(), requires_grad=True).to(device)
        self.u_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_u['u_data'], 1))).float(), requires_grad=True).to(device)


        # boundary points (condition : upper bound = lower bound)
        self.x_lb = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['x_data_lb'], 1))).float(), requires_grad=True).to(device)
        self.t_lb = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['t_data_lb'], 1))).float(), requires_grad=True).to(device)
        self.x_ub = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['x_data_ub'], 1))).float(), requires_grad=True).to(device)
        self.t_ub = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['t_data_ub'], 1))).float(), requires_grad=True).to(device)
