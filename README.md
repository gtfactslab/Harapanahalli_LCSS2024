# Harapanahalli L-CSS 2024 Submission
The paper is available on [arXiv](https://arxiv.org/abs/2309.09043).
## Clone the Repo and its Submodules
```
git clone --recurse-submodules https://github.com/gtfactslab/Harapanahalli_LCSS2024.git
cd Harapanahalli_LCSS2024
```

## Installing Everything into a Conda Environment
```
conda create -n invariance python=3.10
conda activate invariance
```
<!-- Install Pytorch according to [https://pytorch.org/](https://pytorch.org/). If you're using CUDA, check to make sure your CUDA version matches your nvidia driver with `nvidia-smi`. -->

Install `auto_LiRPA` (information taken from [https://github.com/Verified-Intelligence/auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)).
```
cd auto_LiRPA
python setup.py install
```
<!-- If you want their native CUDA modules (CUDA toolkit required),
```
python auto_LiRPA/cuda_utils.py install
``` -->

Install `npinterval`.
```
cd ../npinterval
pip install .
```

Install `gurobi` from [https://www.gurobi.com/](https://www.gurobi.com/).

Step back into the root folder and install the `invariance` package and its dependencies.
```
cd ..
pip install -e .
```

## Leader-Follower Example
### Reproducing the Figure

<!-- You may need `LaTeX` and the following font packages to generate the fonts on the figures correctly.
```
sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng
```
If you don't wish to install `LaTeX`, then you can simply remove lines 139-143 in `example.py`. -->

The following generates Figure 1 as `figures/monotone.pdf`, and also generates the video `figures/monotone.mp4` if you have `ffmpeg` installed, which can be used to visualize the trajectory of the embedding system:
```
cd example
python example.py --notex
```
Use `--notex` if you do not have `LaTeX` installed. Otherwise, you may need to install `LaTeX` and some font packages to properly render the figures.

### Accurate Runtime Estimation

The following runs the same code without plotting as it runs, yielding a more accurate estimation of the runtime.
```
python example.py --runtime --notex
```

<!-- To reproduce the figures from the paper, run the following, where `{model}` is replaced with either `doubleintegrator` , `runtime_N` specifies the number of runs to average over. This can take a while for large values of N. -->

<!-- ## Reproducing Figures from L4DC 2023 Submission

The extended version with proofs is available on [arXiv](https://arxiv.org/abs/2301.07912).

To reproduce the figures from the paper, run the following, where `runtime_N` specifies the number of runs to average over. This can take a while for large values of N.
```
cd examples/vehicle
python vehicle.py --runtime_N 1
```
```
cd examples/quadrotor
python quadrotor.py --runtime_N 1
``` -->
