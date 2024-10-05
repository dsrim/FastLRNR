[![DOI](https://zenodo.org/badge/867897439.svg)](https://doi.org/10.5281/zenodo.13892195)

# FastLRNR 

This is a code repository for the manuscript _FastLRNR and Sparse Physics
Informed Backpropagation_ by Woojin Cho, Kookjin Lee, Noseong Park, Donsub Rim,
and Gerrit Welper.

To reproduce the results of the manuscript run the scripts under the
``./run`` folder in sequence.

1. Convection example
    ```
    cd run \
    && run_train_meta_conv.sh \
    && run_train_test_conv.sh
    ```

1. Convection-diffusion-reaction example
    ```
    cd run \
    && run_train_meta_cdr.sh \
    && run_train_test_cdr.sh
    ```

## LR-PINNs

This code a fork of the [code repository](https://github.com/WooJin-Cho/Hyper-LR-PINN) for the reference:
```
@inproceedings{NEURIPS2023_24f8dd1b,
 author = {Cho, Woojin and Lee, Kookjin and Rim, Donsub and Park, Noseong},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {11219--11231},
 publisher = {Curran Associates, Inc.},
 title = {Hypernetwork-based Meta-Learning for Low-Rank Physics-Informed Neural Networks},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/24f8dd1b8f154f1ee0d7a59e368eccf3-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}
```
