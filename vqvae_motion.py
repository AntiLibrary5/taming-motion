import os
from os.path import join as pjoin
import sys
import json
import logging
import argparse
from argparse import Namespace

import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random
import codecs as cs
from tqdm import tqdm
import pickle
import re


# ARGS
def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--dataname', type=str, default='kit', help='dataset directory')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--window-size', type=int, default=64, help='training motion length')

    ## optimization
    parser.add_argument('--total-iter', default=200000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[50000, 400000], nargs="+", type=int,
                        help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss-vel', type=float, default=0.1, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons-loss', type=str, default='l2', help='reconstruction loss')

    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices=['relu', 'silu', 'gelu'],
                        help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')

    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    parser.add_argument("--resume-gpt", type=str, default=None, help='resume pth for GPT')

    ## output directory
    parser.add_argument('--out-dir', type=str, default='output_vqfinal/', help='output directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug',
                        help='name of the experiment, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--eval-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')

    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')

    return parser.parse_args()


# DATALOADER

POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
}

Loc_list = ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise', 'forward', 'back', 'backward',
            'up', 'down', 'straight', 'curve')

Body_list = ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg', 'waist', 'eye', 'knee', 'shoulder', 'thigh')

Obj_List = ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball', 'handrail', 'baseball', 'basketball')

Act_list = ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat', 'throw', 'hop', 'dance', 'jump', 'turn',
            'stumble', 'dance', 'stop', 'sit', 'lift', 'lower', 'raise', 'wash', 'stand', 'kneel', 'stroll',
            'rub', 'bend', 'balance', 'flap', 'jog', 'shuffle', 'lean', 'rotate', 'spin', 'spread', 'climb')

Desc_list = ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily',
             'angrily', 'sadly')

VIP_dict = {
    'Loc_VIP': Loc_list,
    'Body_VIP': Body_list,
    'Obj_VIP': Obj_List,
    'Act_VIP': Act_list,
    'Desc_VIP': Desc_list,
}


class WordVectorizer(object):
    def __init__(self, meta_root, prefix):
        vectors = np.load(pjoin(meta_root, '%s_data.npy'%prefix))
        words = pickle.load(open(pjoin(meta_root, '%s_words.pkl'%prefix), 'rb'))
        self.word2idx = pickle.load(open(pjoin(meta_root, '%s_idx.pkl'%prefix), 'rb'))
        self.word2vec = {w: vectors[self.word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos):
        pos_vec = np.zeros(len(POS_enumerator))
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator['OTHER']] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, item):
        word, pos = item.split('/')
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._get_pos_ohot(vip_pos)
            else:
                pos_vec = self._get_pos_ohot(pos)
        else:
            word_vec = self.word2vec['unk']
            pos_vec = self._get_pos_ohot('OTHER')
        return word_vec, pos_vec


class VQMotionDataset(data.Dataset):
    def __init__(self, window_size=64, unit_length=4, debug=False, overfit=False):
        self.window_size = window_size
        self.unit_length = unit_length

        self.data_root = '/media/varora/LaCie/Datasets/HumanML3D/HumanML3D/'
        self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
        self.text_dir = pjoin(self.data_root, 'texts')
        self.joints_num = 22
        self.max_motion_length = 196
        self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))

        split_file = pjoin(self.data_root, 'train.txt')

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        assert int(debug) + int(overfit) == 1, "Use either debug mode or overfit mode, not both."
        if debug:
            id_list = id_list[:200]
        if overfit:
            id_list = [id_list[1] for i in range(len(id_list))]

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if motion.shape[0] < self.window_size:
                    continue
                self.lengths.append(motion.shape[0] - self.window_size)
                self.data.append(motion)
            except:
                pass  # missing motion

        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def compute_sampling_prob(self):
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]

        idx = random.randint(0, len(motion) - self.window_size)

        motion = motion[idx:idx + self.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


def motion_dataloader(
               batch_size,
               num_workers=8,
               window_size=64,
               unit_length=4,
               debug=False,
               overfit=False
):
    trainSet = VQMotionDataset(window_size=window_size, unit_length=unit_length, debug=debug, overfit=overfit)
    #prob = trainSet.compute_sampling_prob()
    #sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples=len(trainSet) * 1000, replacement=True)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                               batch_size,
                                               shuffle=True,
                                               # sampler=sampler,
                                               num_workers=num_workers,
                                               # collate_fn=collate_fn,
                                               drop_last=True)
    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


# MODEL
class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * torch.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None):
        super().__init__()
        padding = dilation
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)

        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0, )

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None):
        super().__init__()

        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm)
                  for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = args.mu
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):

        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                                  keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()  # (N, DIM, T)

        return x_d, commit_loss, perplexity


class VQVAE_251(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        self.encoder = Encoder(251 if args.dataname == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(251 if args.dataname == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)
        self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx

    def forward(self, x):

        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        ## quantization
        x_quantized, loss, perplexity = self.quantizer(x_encoder)

        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class HumanVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_251(args, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm)

    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x)  # (N, T)
        return quants

    def forward(self, x):
        x_out, loss, perplexity = self.vqvae(x)

        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out


def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = (value == 'True')
                #     print(key, value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    # print(opt)
    opt_dict['which_epoch'] = 'finest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if opt.dataset_name == 't2m':
        opt.data_root = './dataset/HumanML3D/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = 263
        opt.max_motion_length = 196
        opt.max_motion_frame = 196
        opt.max_motion_token = 55
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        opt.dim_pose = 251
        opt.max_motion_length = 196
        opt.max_motion_frame = 196
        opt.max_motion_token = 55
    else:
        raise KeyError('Dataset not recognized')

    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    return opt


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints):
        super(ReConsLoss, self).__init__()

        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss()

        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4

    def forward(self, motion_pred, motion_gt):
        loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])
        return loss

    def forward_vel(self, motion_pred, motion_gt):
        loss = self.Loss(motion_pred[..., 4: (self.nb_joints - 1) * 3 + 4],
                         motion_gt[..., 4: (self.nb_joints - 1) * 3 + 4])
        return loss


if '__main__' == __name__:
    # TODO: add valdiation/eval code from: https://github.com/Mael-zys/T2M-GPT/blob/main/train_vq.py

    args = get_args_parser()
    torch.manual_seed(args.seed)

    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok=True)

    # logger
    logger = get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    #w_vectorizer = WordVectorizer('./glove', 'our_vab')

    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22

    logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    #eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # dataloader
    train_loader = motion_dataloader(
        args.batch_size,
        window_size=args.window_size,
        unit_length=2 ** args.down_t,
        debug=False,
        overfit=True
    )

    train_loader_iter = cycle(train_loader)

    #val_loader = dataset_TM_eval.DATALoader(args.dataname, False,
    #                                        32,
    #                                        w_vectorizer,
    #                                        unit_length=2 ** args.down_t)

    # network
    net = HumanVQVAE(
       args,  ## use args to define different parameters in different quantizers
       args.nb_code,
       args.code_dim,
       args.output_emb_width,
       args.down_t,
       args.stride_t,
       args.width,
       args.depth,
       args.dilation_growth_rate,
       args.vq_act,
       args.vq_norm
    )

    if args.resume_pth:
        logger.info('loading checkpoint from {}'.format(args.resume_pth))
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        net.load_state_dict(ckpt['net'], strict=True)

    net.train()
    net.cuda()

    # optimizer and scheduler
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    # loss
    Loss = ReConsLoss(args.recons_loss, args.nb_joints)

    # warm-up
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
    for nb_iter in range(1, args.warm_up_iter):

        optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

        gt_motion = next(train_loader_iter)
        gt_motion = gt_motion.cuda().float()  # (bs, 64, dim)

        pred_motion, loss_commit, perplexity = net(gt_motion)
        loss_motion = Loss(pred_motion, gt_motion)
        loss_vel = Loss.forward_vel(pred_motion, gt_motion)

        loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()

        if nb_iter % args.print_iter == 0:
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter

            logger.info(
                f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")

            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

    # train
    avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
    for nb_iter in range(1, args.total_iter + 1):
        gt_motion = next(train_loader_iter)
        gt_motion = gt_motion.cuda().float()  # bs, nb_joints, joints_dim, seq_len

        pred_motion, loss_commit, perplexity = net(gt_motion)
        loss_motion = Loss(pred_motion, gt_motion)
        loss_vel = Loss.forward_vel(pred_motion, gt_motion)

        loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()

        if nb_iter % args.print_iter == 0:
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter

            writer.add_scalar('./Train/L1', avg_recons, nb_iter)
            writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
            writer.add_scalar('./Train/Commit', avg_commit, nb_iter)

            logger.info(
                f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")

            avg_recons, avg_perplexity, avg_commit = 0., 0., 0.,

