import argparse
import os
import sys
import numpy as np


parser = argparse.ArgumentParser()
# CPU / GPU setting


parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--use_cuda', type=str, default='True')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--start_coeff_1', type=int, default=0, help='start point of beta range')
parser.add_argument('--start_coeff_2', type=int, default=0, help='start point of nu range')
parser.add_argument('--start_coeff_3', type=int, default=0, help='start point of rho range')

parser.add_argument('--end_coeff_1', type=int, default=0, help='end point of beta range')
parser.add_argument('--end_coeff_2', type=int, default=0, help='end point of nu range')
parser.add_argument('--end_coeff_3', type=int, default=0, help='end point of rho range')

parser.add_argument('--init_cond', type=str, default='sin_1')
parser.add_argument('--pde_type', type=str, default='convection')

parser.add_argument('--target_coeff_1', type=int, default=0, help='target coefficient beta')
parser.add_argument('--target_coeff_2', type=int, default=0, help='target coefficient nu')
parser.add_argument('--target_coeff_3', type=int, default=0, help='target coefficient rho')
parser.add_argument('--reg_param', type=float, default=0, help='regularization parameter')
parser.add_argument('--eim_rank', type=int, default=0, help='rank of eim basis')
parser.add_argument('--npts', type=int, default=0, help='no of sample pts')
parser.add_argument('--hidden_dim', type=int, default=50, help='hidden_dim')
parser.add_argument('--alpha_dim', type=int, default=50, help='alpha_dim')
parser.add_argument('--learn_rate', type=float, default=0.001, help='learn_rate')

parser.add_argument('--epoch', type=int, default=10000)
parser.add_argument('--load_epoch', type=int, default=20000)
def get_config():
    return parser.parse_args()
