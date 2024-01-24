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

### Using SMPL models

Since the SMPL fmaily of models require a licence agrrement, there is no automatic script to download them. Please follow the instructions in the [smplx](https://github.com/vchoutas/smplx#loading-smpl-x-smplh-and-smpl) repository to do so.

NB: If you have access to a local inria machine (with acess to the LaaCie data depot) a ready to use folder can be found in **/home/adakri/varora_lacies/LaCie/Models**.

### TO-DO
You will need to either download and process your own copy of HumanML3D or mount:
```angular2html
'/media/varora/LaCie/Datasets/HumanML3D/HumanML3D/'
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade --force-reinstall

## TO-DO
- [x] Download HumanML3D dataset
- [x] Visualize HumanML3D dataset
- [x] Overfit a VQVAE codebook on a single sample
- [ ] Visualize reconstructions of overfitted VQVAE
- [x] Train a VQVAE on whole HumanML3D dataset
- [x] Visualize reconstructions of model trained on whole HumanML3D dataset
- [ ] Train a masked autoencoder on top of the learnt motion codebook

# Results/Updates
```angular2html
https://gitlab.inria.fr/varora/taming-motion/-/issues/2
```
## Notes
- Issue:
    ```angular2html
    ModuleNotFoundError: No module named 'mpl_toolkits'
    ```
    Solution:
    ```angular2html
    pip install basemap
    ```
- Issue
  ```angular2html
  ax.lines = []
  AttributeError: can't set attribute
  ```
  Solution:
  ```angular2html
  conda env create -f environment_t2m.yml 
  conda activate T2M-GPT
  ```
  
## Dataset
Using the [HumanML3D](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.pdf) 
data: a 3D human motion-language dataset that originates from a combination of HumanAct12 and Amass dataset. Follow the 
instructions in the [HumanML3D github repo](https://github.com/EricGuo5513/HumanML3D) to download and process the dataset.

Data directory: `/media/varora/LaCie1/Datasets/HumanML3D/`

<img src="assets/data/gifs/000000.gif" width="25%" height="25%"/>
<img src="assets/data/gifs/000001.gif" width="25%" height="25%"/>
<img src="assets/data/gifs/000002.gif" width="25%" height="25%"/>
<img src="assets/data/gifs/000003.gif" width="25%" height="25%"/>
<img src="assets/data/gifs/000004.gif" width="25%" height="25%"/>
<img src="assets/data/gifs/000005.gif" width="25%" height="25%"/>
<img src="assets/data/gifs/000006.gif" width="25%" height="25%"/>
<img src="assets/data/gifs/000007.gif" width="25%" height="25%"/>
<img src="assets/data/gifs/000008.gif" width="25%" height="25%"/>

## Train VQVAE
```angular2html
python vqvae_motion.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir output --dataname t2m --vq-act relu --loss-vel 0.5 --recons-loss l1_smooth --exp-name motion-vqvae
```
Trained model: https://mybox.inria.fr/f/ef138f165e51480d8c53/?dl=1

## Eval VQVAE
```angular2html
python vqvae_motion.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir output --dataname t2m --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name motion-vqvae-test --resume-pth output/motion-vqvae/net_last.pth --eval
```

## Train GM3
```angular2html
python gm3.py --mask_ratio 0.6 --batch-size 256 --lr 2e-4 --total-iter 250000 --lr-scheduler 200000 --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir output --dataname t2m --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name gm3-train --resume-pth output/motion-vqvae/net_last.pth
```

## Eval VQVAE with mask tokens
Download model from: https://mybox.inria.fr/f/9ce3b9c67ec344a2a97f/?dl=1
Place it as: `output/motion-vqvae-with-mask-token/net_last.pth`
```angular2html
python gm3.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir output --dataname t2m --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name motion-vqvae-with-mask-test-gifs --resume-pth output/motion-vqvae-with-mask-token/net_last.pth --with_mask_token
```

## Visualize Results
```angular2html
tensorboard --logdir=<EXP-NAME>
```