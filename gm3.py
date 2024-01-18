import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim

from vqvae_motion import (get_args_parser, get_logger, WordVectorizer, get_opt, EvaluatorModelWrapper,
                          eval_dataloader, HumanVQVAE, motion_dataloader, cycle)

from mmm import MMM

args = get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok=True)

##### ---- Logger ---- #####
logger = get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

w_vectorizer = WordVectorizer('./glove', 'our_vab')

dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Dataloader ---- #####
args.nb_joints = 22

val_loader = eval_dataloader(args.dataname, True, 32, w_vectorizer, unit_length=2 ** args.down_t)

##### ---- Network ---- #####
vqvae = HumanVQVAE(args,  ## use args to define different parameters in different quantizers
                 args.nb_code,
                 args.code_dim,
                 args.output_emb_width,
                 args.down_t,
                 args.stride_t,
                 args.width,
                 args.depth,
                 args.dilation_growth_rate,
                 args.vq_act,
                 args.vq_norm)

if args.resume_pth:
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    vqvae.load_state_dict(ckpt['net'], strict=True)

vqvae.eval()
vqvae.cuda()

m3 = MMM()
m3.train()
m3.cuda()

code_dim = 512

# optimizer and scheduler
optimizer = optim.AdamW(m3.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

train_loader = motion_dataloader(
    args.batch_size,
    window_size=args.window_size,
    unit_length=2 ** args.down_t,
    debug=False,
    overfit=False
)
train_loader_iter = cycle(train_loader)

avg_recons_loss = 0.
for nb_iter in range(1, args.total_iter + 1):
    motion = next(train_loader_iter)
    motion = motion.cuda().float()

    pred_pose, loss_commit, perplexity = vqvae(motion)
    code_idx = vqvae.encode(motion)
    quants = vqvae.vqvae.quantizer.dequantize(code_idx)
    #quants = quants.view(1, -1, code_dim).permute(0, 2, 1).contiguous()

    logits, mask, target = m3(quants)

    loss = F.mse_loss(
        logits,
        quants,
        reduction="mean",
    )

    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    optimizer.zero_grad(set_to_none=True)

    avg_recons_loss += loss.item()

    if nb_iter % args.print_iter == 0:
        avg_recons_loss /= args.print_iter
        writer.add_scalar('./Train/MES_loss', avg_recons_loss, nb_iter)
        logger.info(
            f"Step: {nb_iter + 1} "
            f"Loss: {avg_recons_loss:0.4f} "
            f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
        )