
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

    lrnr = lrnr.LRNR(pde_type="cdr",
                     reg_param=1e-4,
                     start_coeff_1 = 1,
                     end_coeff_1 = 3,
                     start_coeff_2 = 0,
                     end_coeff_2 = 2,
                     start_coeff_3 = 0,
                     end_coeff_3 = 2,
                     )
    lrnr.learn_rate = 1e-4 / 2

    coeff_1_tensor = torch.linspace(1, 3, 6)
    coeff_2_tensor = torch.linspace(0, 2, 6)
    coeff_3_tensor = torch.linspace(0, 2, 6)

    C1,C2,C3 = torch.meshgrid(coeff_1_tensor, coeff_2_tensor, coeff_3_tensor, indexing="ij")
    coeff_tensor = torch.cat([C1.reshape(-1, 1),
                              C2.reshape(-1, 1),
                              C3.reshape(-1, 1)], dim=1)

    lrnr.compute_truncated_alpha_range(coeff_tensor)
    lrnr.set_uniform_eim_pts()
    npts = len(lrnr.x_collocation_fast)

    coeff0 = torch.tensor([3.0, 0.2, 2.0])
    lrnr.load_phase2_target(coeff0)
    net0 = lrnr.net_phase2

    nx = 256
    nt = 100
    x_tensor = torch.linspace(0.0, 2.0*torch.pi, nx)
    t_tensor = torch.linspace(0.0, 1.0, nt)
    X, T = torch.meshgrid(x_tensor, t_tensor, indexing="ij")
    #XT = torch.cat([X.reshape(-1, 1), T.reshape(-1, 1)], dim=1)
    Z = net0(X.reshape(-1,1), T.reshape(-1, 1))

    X_array = X.detach().numpy()
    T_array = T.detach().numpy()
    Z_array = Z.detach().numpy().reshape(nx,nt)

    u_test = lrnr.compute_numerical_solution(coeff0)

    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    ax = axs[0]
    ax.pcolormesh(X_array, T_array, Z_array)
    ax = axs[1]
    ax.pcolormesh(X_array, T_array, u_test.reshape(nt,nx).T)
    ax = axs[2]
    im = ax.pcolormesh(X_array, T_array, u_test.reshape(nt,nx).T - Z_array)
    plt.colorbar(im)
    fig.show()

    coeff_1_tensor = torch.linspace(1, 3, 4)
    coeff_2_tensor = torch.linspace(0, 2, 4)
    coeff_3_tensor = torch.linspace(0, 2, 4)
    eim_C1,eim_C2,eim_C3 = \
        torch.meshgrid(coeff_1_tensor, coeff_2_tensor, coeff_3_tensor, indexing="ij")

    eim_sample_coeff_tensor = torch.cat([eim_C1.reshape(-1, 1),
                                         eim_C2.reshape(-1, 1),
                                         eim_C3.reshape(-1, 1)], dim=1)

    print("computing eim basis:", len(eim_sample_coeff_tensor), "npts =", npts)

    lrnr.compute_eim(eim_sample_coeff_tensor, npts=npts)

    ntargets = 10

    target_coeff_1_tensor = 2.0*torch.rand(ntargets) + 1.0
    target_coeff_2_tensor = 2.0*torch.rand(ntargets)
    target_coeff_3_tensor = 2.0*torch.rand(ntargets)

    target_coeff_tensor = torch.cat([target_coeff_1_tensor.reshape(-1, 1),
                                     target_coeff_2_tensor.reshape(-1, 1),
                                     target_coeff_3_tensor.reshape(-1, 1)], dim=1)


    if not os.path.exists("_plot"): os.mkdir("_plot")

    # start main loop
    counter = 0
    lrnr.compute_eim(eim_sample_coeff_tensor, npts=npts)

    reg_param_array = [
                        0.30,
                        1.48,
                        0.40,
                        0.14,
                        0.48,
                        0.30,
                        2.74,
                        1.56,
                        0.14,
                        0.22,
                        ]

    eim_rank = 5
    min_err_array = np.zeros(len(target_coeff_tensor))

    save_path = './log/phase2/cdr/sin_1'

    np.savetxt("targets.txt", target_coeff_tensor.detach().numpy())

    for i, target_tensor in enumerate(target_coeff_tensor):

        reg_param = reg_param_array[i]

        plot_path = os.path.join("_plot",
                                "npts{:02d}_rank{:02d}".format(npts, eim_rank))
        if not os.path.exists(plot_path): os.mkdir(plot_path)

        lrnr.eim_rank = eim_rank
    
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
            ["ti{:02d}".format(i),
             "r{:08.4f}".format(reg_param).replace(".","-")])

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
