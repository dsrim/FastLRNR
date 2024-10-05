
import os, sys
import numpy as np
import torch
import numpy as np

import lrnr

# plotting
import matplotlib
matplotlib.rcParams["font.family"] = "monospace"
import matplotlib.pyplot as plt
matplotlib.style.use("dark_background")
import itertools


if __name__ == "__main__":

    device_train_adap = "cuda"
    device_train_fast_test = "cuda"

    lrnr = lrnr.LRNR()

    # set coefficients to compute nonzero entries
    coeff_min = 5.0
    coeff_max = 8.0
    n = 20

    coeff1_range = torch.linspace(coeff_min, coeff_max, n).reshape(-1, 1)
    coeff_tensor = torch.cat([
                            coeff1_range,
                            torch.zeros_like(coeff1_range), 
                            torch.zeros_like(coeff1_range),
                            ], dim=1)

    lrnr.compute_truncated_alpha_range(coeff_tensor)

    lrnr.set_uniform_eim_pts()

    # sample EIM coefficients
    n = 500
    coeff1_range = torch.linspace(coeff_min, coeff_max, n).reshape(-1, 1)
    eim_sample_coeff_tensor = torch.cat([
                                    coeff1_range,
                                    torch.zeros_like(coeff1_range), 
                                    torch.zeros_like(coeff1_range),
                                    ], dim=1)

    npts = len(lrnr.x_collocation_fast)

    # set target coeffs
    coeff_min = 5.0
    coeff_max = 8.0
    ntargets = 10
    coeff1_tensor = \
    ((coeff_max - coeff_min)*torch.rand(ntargets) + coeff_min).reshape(-1, 1)
    #print(coeff1_tensor)
    
    target_coeff_tensor = torch.cat([
                                coeff1_tensor,
                                torch.zeros_like(coeff1_tensor),
                                torch.zeros_like(coeff1_tensor)
                                ], dim=1)

    reg_param_array = np.array([
                                 1.10,
                                 1.24,
                                 1.02,
                                 1.02,
                                 1.18,
                                17.60,
                                 1.00,
                                 1.02,
                                 1.02,
                                12.30,
                                ])
    min_err_array = np.zeros(len(target_coeff_tensor))

    if not os.path.exists("_plot"): os.mkdir("_plot")


    # start main loop
    lrnr.learn_rate = 1e-4 / 2
    counter = 0
    lrnr.compute_eim(eim_sample_coeff_tensor, npts=npts)
    eim_rank = 5

    lrnr.eim_rank = eim_rank

    plot_path = os.path.join("_plot",
                                "npts{:02d}_rank{:02d}".format(npts, eim_rank))
    save_path = './log/phase2/convection/sin_1'

    if not os.path.exists(plot_path): os.mkdir(plot_path)

    for target in enumerate(target_coeff_tensor):

        i, target_tensor = target
        reg_param = reg_param_array[i]
    
        # run fast lrnr
        lrnr.train_fast(target_tensor,
                        reg_param=reg_param,
                        test_device=device_train_fast_test,
                        epoch=400,)

        # run train adap
        lrnr.train_adap(target_tensor,
                        device=device_train_adap,
                        epoch=400,)

        target_str = lrnr.get_target_str(target_tensor)
        save_str = "_".join(
        ["ti{:02d}".format(i), "r{:08.4f}".format(reg_param).replace(".","-")])


        ## Create plots, store data

        data_fname = \
            os.path.join(save_path,"PINN_adap_{:s}.npy".format(target_str))
        data_adap = np.load(data_fname)
        data_adap = data_adap.T    # TODO: adjust dims..

        data_fname = \
            os.path.join(save_path,"PINN_fast_{:s}.npy".format(target_str))
        data_fast = np.load(data_fname)
        data_fast = data_fast.T

        sol_fname = \
            os.path.join(save_path,"PINN_adap_sol_{:s}.npy".format(target_str))
        sol_adap = np.load(sol_fname).T

        sol_fname = \
            os.path.join(save_path,"PINN_fast_sol_{:s}.npy".format(target_str))
        sol_fast = np.load(sol_fname).T
        
        # make error plot
        fig, ax = plt.subplots()
        ax.set_title(target_str.replace("-", "."))
        ax.semilogy(data_adap[:, :2], linestyle="--",
                    label=["L1 adap", "L2 adap"])
        ax.semilogy(data_fast[:, :2], label=["L1 fast", "L2 fast"])
        ax.legend()
        ax.set_xlabel("epoch")
        ax.set_ylabel("rel err")
        plot_fpath = \
            os.path.join(plot_path, "errors_{:s}.pdf".format(save_str))
        fig.savefig(plot_fpath)
        plt.close(fig)

        # make loss plot
        fig, axs = plt.subplots(nrows=2)
        ax = axs[0]
        ax.set_title(target_str.replace("-", "."))
        ax.semilogy(data_adap[:, 2], linestyle="--",
                    label="loss adap")
        ax.legend()
        ax = axs[1]
        ax.semilogy(data_fast[:, 2], label="loss fast")
        ax.set_xlabel("epoch")
        ax.legend()
        plot_fpath = \
            os.path.join(plot_path, "losses_{:s}.pdf".format(save_str))
        fig.savefig(plot_fpath)
        plt.close(fig)

        # solution plot
        nt = 100
        nx = 256
        x_test = sol_adap[:, 0].reshape(nt, nx)
        t_test = sol_adap[:, 1].reshape(nt, nx)
        u_adap = sol_adap[:, 2].reshape(nt, nx)
        u_fast = sol_fast[:, 2].reshape(nt, nx)

        fig, axs = plt.subplots(
                                ncols=3,
                                figsize=(15, 5),
                                layout="constrained",
                                )
        ax = axs[0]
        ax.pcolormesh(x_test, t_test, u_adap)

        ax = axs[1]
        im = ax.pcolormesh(x_test, t_test, u_fast)

        ax = axs[2]
        vmax = np.abs(u_fast - u_adap).flatten().max()
        im = ax.pcolormesh(x_test, t_test, u_fast - u_adap,
                           vmax=vmax, vmin=-vmax, cmap="RdBu")
        plt.colorbar(im, ax=ax)

        plot_fpath = \
            os.path.join(plot_path, "sol_{:s}.pdf".format(save_str))
        fig.savefig(plot_fpath, bbox_inches="tight")
        plt.close(fig)
        

        # store min error
        min_err = np.min(data_fast[:, 1])
        print("="*80)
        print("     {:1.4f},fast min L2 err:  {:1.4e}".format(reg_param, min_err))
        min_err_array[i] = min_err
        np.savetxt("reg_sweep_min_err.txt", min_err_array)

        # store adap / fast error
        plot_data_fpath = \
            os.path.join(plot_path, "data_{:s}.txt".format(save_str))
        np.savetxt(plot_data_fpath, np.hstack([data_adap, data_fast]))
