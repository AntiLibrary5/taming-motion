# Taming Motion
Trying an idea of learning a motion codebook from a dataset of human motion and learning the distribution of the
codebook using a masked auto-encoding approach compared to an autoregressive transformer based approach.

## Requirements
Create and activate a [conda](https://conda.io/) environment as follows:
```
conda env create -f environment.yaml
conda activate taming-motion
pip install git+https://github.com/nghorbani/body_visualizer.git
pip install git+https://github.com/MPI-IS/configer
pip install git+https://github.com/MPI-IS/mesh.git
pip install -e .
```
Note: ensure `libboost-dev`, `gcc`, `g++` are installed on your system.

### TO-DO
[ x ] Download HumanML3D dataset
[ ] Train a VQ-GAN codebook and visualize reconstructions
[ ] Train a masked autoencoder on top of the learnt motion codebook