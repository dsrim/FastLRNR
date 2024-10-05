import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from config import get_config
import torch
import random
import torch.backends.cudnn as cudnn
import pandas as pd
from model import LR_PINN_phase1
from utils import orthogonality_reg, f_cal, get_params
import os, sys
from sklearn.metrics import explained_variance_score, max_error

args = get_config()
device = torch.device(args.device)

def main():
    args=get_config()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    
    device = torch.device(args.device)

    print("=============[Deivce Info]==============")
    print("- Use Device :", device)
    print("- Available cuda devices :", torch.cuda.device_count())
    print("- Current cuda device :", torch.cuda.current_device())
    print("- Name of cuda device :", torch.cuda.get_device_name(device))
    print("========================================\n")

    mse_cost_function = torch.nn.MSELoss() # Mean squared error
    hidden_dim = args.hidden_dim
    alpha_dim = args.alpha_dim
    reg_param = args.reg_param

    net = LR_PINN_phase1(hidden_dim, reg_param=reg_param, alpha_dim=alpha_dim,)
                                    
    net = net.to(device)

    model_size = get_params(net)
    
    #######################################
    ############   argparser   ############
    epoch = args.epoch

    initial_condition = args.init_cond
    pde_type = args.pde_type  
    
    start_coeff_1 = args.start_coeff_1
    start_coeff_2 = args.start_coeff_2
    start_coeff_3 = args.start_coeff_3

    end_coeff_1 = args.end_coeff_1
    end_coeff_2 = args.end_coeff_2
    end_coeff_3 = args.end_coeff_3
    
    #######################################
    #######################################

    beta = start_coeff_1
    nu = start_coeff_2
    rho = start_coeff_3

    train_boundary_fname = './data_gen/dataset/cdr/train/train_boundary_' + str(int(beta)) + '_' + str(int(nu)) + '_' + str(int(rho)) + '_cdr.csv'
    train_u_fname        = './data_gen/dataset/cdr/train/train_u_' + str(int(beta)) + '_' + str(int(nu)) + '_' + str(int(rho)) + '_cdr.csv'
    train_f_fname        = './data_gen/dataset/cdr/train/train_f_' + str(int(beta)) + '_' + str(int(nu)) + '_' + str(int(rho)) + '_cdr.csv'
    test_fname           = './data_gen/dataset/cdr/test/test_' + str(int(beta)) + '_' + str(int(nu)) + '_' + str(int(rho)) + '_cdr.csv'

    train_data_f    = pd.read_csv(train_f_fname)
    train_data_u    = pd.read_csv(train_u_fname)
    train_data_bd   = pd.read_csv(train_boundary_fname)
    test_data       = pd.read_csv(test_fname)

    for beta in range(start_coeff_1, end_coeff_1+1):
        for nu in range(start_coeff_2, end_coeff_2+1):
            for rho in range(start_coeff_3, end_coeff_3+1):

                train_boundary_fname = './data_gen/dataset/cdr/train/train_boundary_' + str(int(beta)) + '_' + str(int(nu)) + '_' + str(int(rho)) + '_cdr.csv'
                train_u_fname        = './data_gen/dataset/cdr/train/train_u_' + str(int(beta)) + '_' + str(int(nu)) + '_' + str(int(rho)) + '_cdr.csv'
                train_f_fname        = './data_gen/dataset/cdr/train/train_f_' + str(int(beta)) + '_' + str(int(nu)) + '_' + str(int(rho)) + '_cdr.csv'
                test_fname           = './data_gen/dataset/cdr/test/test_' + str(int(beta)) + '_' + str(int(nu)) + '_' + str(int(rho)) + '_cdr.csv'

                f_sample    = pd.read_csv(train_f_fname)
                u_sample    = pd.read_csv(train_u_fname)
                bd_sample   = pd.read_csv(train_boundary_fname)
                test_sample = pd.read_csv(test_fname)

                train_data_f    = pd.concat([train_data_f, f_sample], ignore_index = True)
                train_data_u    = pd.concat([train_data_u, u_sample], ignore_index = True)
                train_data_bd   = pd.concat([train_data_bd, bd_sample], ignore_index = True)
                test_data       = pd.concat([test_data, test_sample], ignore_index = True)

    x_collocation = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_f['x_data'], 1))).float(), requires_grad=True).to(device)
    t_collocation = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_f['t_data'], 1))).float(), requires_grad=True).to(device)
    beta_collocation = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_f['beta'], 1))).float(), requires_grad=True).to(device)
    nu_collocation = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_f['nu'], 1))).float(), requires_grad=True).to(device)
    rho_collocation = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_f['rho'], 1))).float(), requires_grad=True).to(device)


    # initial points
    x_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_u['x_data'], 1))).float(), requires_grad=True).to(device)
    t_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_u['t_data'], 1))).float(), requires_grad=True).to(device)
    u_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_u['u_data'], 1))).float(), requires_grad=True).to(device)
    beta_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_u['beta'], 1))).float(), requires_grad=True).to(device)
    nu_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_u['nu'], 1))).float(), requires_grad=True).to(device)
    rho_initial = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_u['rho'], 1))).float(), requires_grad=True).to(device)
    

    # boundary points (condition : upper bound = lower bound)
    x_lb = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['x_data_lb'], 1))).float(), requires_grad=True).to(device)
    t_lb = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['t_data_lb'], 1))).float(), requires_grad=True).to(device)
    x_ub = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['x_data_ub'], 1))).float(), requires_grad=True).to(device)
    t_ub = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['t_data_ub'], 1))).float(), requires_grad=True).to(device)
    beta_bd = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['beta'], 1))).float(), requires_grad=True).to(device)
    nu_bd = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['nu'], 1))).float(), requires_grad=True).to(device)
    rho_bd = Variable(torch.from_numpy(np.array(np.expand_dims(train_data_bd['rho'], 1))).float(), requires_grad=True).to(device)

    
    # test point 
    x_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['x_data'], 1))).float(), requires_grad=False).to(device)
    t_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['t_data'], 1))).float(), requires_grad=False).to(device)
    u_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['u_data'], 1))).float(), requires_grad=False).to(device)
    beta_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['beta'], 1))).float(), requires_grad=False).to(device)
    nu_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['nu'], 1))).float(), requires_grad=False).to(device)
    rho_test = Variable(torch.from_numpy(np.array(np.expand_dims(test_data['rho'], 1))).float(), requires_grad=False).to(device)
         
    print("=============[Train Info]===============")
    print(f"- PDE type : {pde_type}")
    print(f"- Initial condition : {initial_condition}")
    print(f"- start_coeff_1 ~ end_coeff_1 :{start_coeff_1} ~ {end_coeff_1}")
    print(f"- Model size : {model_size}")
    print("========================================\n")

    print("=============[Model Info]===============\n")
    print(net)
    print("========================================\n")

    lr0=1e-3
    optimizer = optim.Adam(net.parameters(), lr=lr0)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer,"min",
                     factor=0.95, min_lr=1e-6, patience=100, threshold=1e-3)
    
    err_list = []
    ep_list = []
    loss_list= []

    mse_loss_list = []
    reg_loss_list = []

    mse_u_list = []
    mse_f_list = []
    mse_bd_list = []
    
    reg_u_list = []
    reg_f_list = []
    reg_bd_list = []

    L2_abs_list = []
    L2_rel_list = []
    Max_err_list = []
    Ex_var_score_list = []

    nc = 4000
    nt = 20 # divisor of 1000

    for ep in range(1, epoch+1):
        net.train()

        optimizer.zero_grad()

        # compute initial condition loss
        out = net(
                  x_initial,
                  t_initial,
                  beta_initial,
                  nu_initial,
                  rho_initial,
                  )        
        [
            net_initial_out,
            col_0_init,
            col_1_init,
            col_2_init,
            row_0_init,
            row_1_init,
            row_2_init
        ] \
        = out
        mse_u = mse_cost_function(net_initial_out, u_initial)

        # compute orthogonality constraints for initial condition
        reg_init_0 = orthogonality_reg(col_0_init, row_0_init, alpha_dim)
        reg_init_1 = orthogonality_reg(col_1_init, row_1_init, alpha_dim)
        reg_init_2 = orthogonality_reg(col_2_init, row_2_init, alpha_dim)

        reg_init = reg_init_0 + reg_init_1 + reg_init_2

        f_out, reg_f = f_cal(
                           x_collocation,
                           t_collocation,
                           beta_collocation,
                           nu_collocation,
                           rho_collocation,
                           net,
                           alpha_dim,
                           ) 

        mse_f = mse_cost_function(f_out, torch.zeros_like(f_out))

        u_pred_lb, col_0_lb, col_1_lb, col_2_lb, row_0_lb, row_1_lb, row_2_lb \
        = net(x_lb, t_lb, beta_bd, nu_bd, rho_bd)
        u_pred_ub, col_0_ub, col_1_ub, col_2_ub, row_0_ub, row_1_ub, row_2_ub \
        = net(x_ub, t_ub, beta_bd, nu_bd, rho_bd)
        
        reg_lb_0 = orthogonality_reg(col_0_lb, row_0_lb, alpha_dim)
        reg_lb_1 = orthogonality_reg(col_1_lb, row_1_lb, alpha_dim)
        reg_lb_2 = orthogonality_reg(col_2_lb, row_2_lb, alpha_dim)

        reg_ub_0 = orthogonality_reg(col_0_ub, row_0_ub, alpha_dim)
        reg_ub_1 = orthogonality_reg(col_1_ub, row_1_ub, alpha_dim)
        reg_ub_2 = orthogonality_reg(col_2_ub, row_2_ub, alpha_dim)

        reg_bd = reg_lb_0 + reg_lb_1 + reg_lb_2 + reg_ub_0 + reg_ub_1 + reg_ub_2

        mse_bd = torch.mean((u_pred_lb - u_pred_ub) ** 2)
                
        loss = mse_u + mse_f + mse_bd + reg_init + reg_f + reg_bd

        loss.backward()
        optimizer.step()
        
        scheduler2.step(loss)

        for g in optimizer.param_groups:
            lr0 = g["lr"]
        
        if ep % 10 == 0:
            net.eval()
            with torch.autograd.no_grad():

                mse_test = 0.0
                L2_error_norm = 0.0
                L2_true_norm = 0.0

                L2_absolute_error = 0.0
                L2_relative_error = 0.0
                Max_err = 0.0
                
                batch_no = len(x_test) // nt

                for i in range(batch_no):
                    ib = (nt*i % len(x_test))
                    ie = (nt*(i+1) % (len(x_test)+1))

                    x_test0    = x_test[ib:ie, :]
                    t_test0    = t_test[ib:ie, :]
                    beta_test0 = beta_test[ib:ie, :]
                    nu_test0   = nu_test[ib:ie, :]
                    rho_test0  = rho_test[ib:ie, :]

                    u_out_test, _, _, _, _, _, _ = net(x_test0,
                                                    t_test0,
                                                    beta_test0,
                                                    nu_test0,
                                                    rho_test0,
                                                    )

                    mse_test += mse_cost_function(u_out_test, u_test[ib:ie, ...])**2
                    L2_error_norm += torch.linalg.norm(u_out_test-u_test[ib:ie, ...], 2, dim = 0)**2 / batch_no
                    L2_true_norm += torch.linalg.norm(u_test[ib:ie, ...], 2, dim = 0) **2 / batch_no
                    L2_absolute_error += torch.mean(torch.abs(u_out_test-u_test[ib:ie, ...])) / batch_no

                    u_test_cpu = u_test[ib:ie, ...].cpu()
                    u_out_test_cpu = u_out_test.cpu()
                    max_batch = max_error(u_test_cpu, u_out_test_cpu)

                    if max_batch > Max_err:
                        Max_err = max_batch

                mse_test = torch.sqrt(mse_test).item()
                L2_error_norm = torch.sqrt(L2_error_norm).item()
                L2_true_norm = torch.sqrt(L2_true_norm).item()
                L2_relative_error = L2_error_norm / L2_true_norm
                L2_absolute_error = L2_absolute_error.item()

                err_list.append(mse_test)
                ep_list.append(ep)
                loss_list.append(loss.item())
                
                mse_loss_list.append((mse_u+mse_f+mse_bd).item())
                reg_loss_list.append((reg_init+reg_f+reg_bd).item())
                
                mse_u_list.append(mse_u.item())
                mse_f_list.append(mse_f.item())
                mse_bd_list.append(mse_bd.item())
                
                reg_u_list.append(reg_init.item())
                reg_f_list.append(reg_f.item())
                reg_bd_list.append(reg_bd.item())
                
                L2_abs_list.append(L2_absolute_error)
                L2_rel_list.append(L2_relative_error)
                Max_err_list.append(Max_err)

                Ex_var_score = 0.0 # kluging, ignoring for now
                Ex_var_score_list.append(0.0)
                
                sys.stdout.write("\n")
                print('-'*30 + " meta training " + '-'*30)
                print('lr : {:1.4e} hdim: {:6d} adim: {:6d}'.format(lr0, hidden_dim, alpha_dim))
                print('reg param : {:1.4e}'.format(net.reg_param.item()))
                print('L2_abs_err : {:1.4f}'.format(L2_absolute_error))
                print('L2_rel_err : {:1.4f}'.format(L2_relative_error))
                print('Max_err :', Max_err)
                print('Ex_var_score :', Ex_var_score)
                print('Epoch : {:8d} Error : {:1.4e} train_loss (total) : {:1.4e}'.format(ep, mse_test,loss.item()))    
                print('mse_f : {:1.4e} mse_u : {:1.4e}  mse_bd : {:1.4e}'.format(mse_f.item(), mse_u.item(),mse_bd.item()))
                print('reg_f : {:1.4e}'.format(reg_f.item()))


        if (ep+1) % 1000 == 0:

            SAVE_PATH = f'./param/phase1/{pde_type}/{initial_condition}'
            SAVE_NAME = f'PINN_{start_coeff_1}_{end_coeff_1}_{ep+1}_h{hidden_dim}' + "_reg{:1.4e}".format(reg_param).replace(".","o") +  '.pt'
            
            if not os.path.isdir(SAVE_PATH): os.mkdir(SAVE_PATH)
            torch.save(net.state_dict(), SAVE_PATH + "/" + SAVE_NAME)

    err_df = pd.DataFrame(err_list)
    ep_df = pd.DataFrame(ep_list)
    loss_df = pd.DataFrame(loss_list)
    
    mse_loss_df = pd.DataFrame(mse_loss_list)
    reg_loss_df = pd.DataFrame(reg_loss_list)
    
    mse_u_df = pd.DataFrame(mse_u_list)
    mse_f_df = pd.DataFrame(mse_f_list)
    mse_bd_df = pd.DataFrame(mse_bd_list)
    
    reg_u_df = pd.DataFrame(reg_u_list)
    reg_f_df = pd.DataFrame(reg_f_list)
    reg_bd_df = pd.DataFrame(reg_bd_list)
    
    L2_abs_df = pd.DataFrame(L2_abs_list)
    L2_rel_df = pd.DataFrame(L2_rel_list)
    Max_err_df = pd.DataFrame(Max_err_list)
    Ex_var_score_df = pd.DataFrame(Ex_var_score_list)
    
    log_data = pd.concat([ep_df, loss_df, err_df, mse_loss_df, reg_loss_df, mse_u_df, mse_f_df, mse_bd_df, reg_u_df, reg_f_df, reg_bd_df, L2_abs_df, L2_rel_df, Max_err_df, Ex_var_score_df], axis=1)
    log_data.columns = ["epoch", "train_loss", "test_err", "mse_loss", "reg_loss", "mse_u", "mse_f", "mse_bd", "reg_u", "reg_f", "reg_bd", "L2_abs_err", "L2_rel_err", "Max_err", "Ex_var_score"]    

    log_path = f'./log/phase1/{pde_type}/{initial_condition}'
    log_name = f'PINN_{start_coeff_1}_{end_coeff_1}_h{hidden_dim}' + '_reg{:1.4e}'.format(reg_param).replace(".","o") + ".csv"
    if not os.path.isdir(log_path): 
        os.mkdir(log_path)
        
    log_data.to_csv(log_path+"/"+log_name, index=False)

    print('#### final ####')
    print('L2_abs_err :', L2_absolute_error)
    print('L2_rel_err :', L2_relative_error)

    print('Epoch :', ep, 'Error :', mse_test, 'train_loss (total) :', loss.item())    
    print('mse_f :', mse_f.item(), 'mse_u :', mse_u.item(), 'mse_bd :', mse_bd.item())
    print('#'*80)


if __name__ == "__main__":
    main()
