import os
import json

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim

from vqvae_motion import (get_args_parser, get_logger, WordVectorizer, get_opt, EvaluatorModelWrapper,
                          eval_dataloader, HumanVQVAE, motion_dataloader, cycle, recover_from_ric,
                          draw_to_batch)

from mmm import MMM

class KeyFrameMask(nn.Module):
    """
    random masking
    """
    def __init__(self, num_patches):
        super().__init__()
        self.num_patches = num_patches
        self.num_mask = int(self.num_patches/2)

    def __call__(self, x):
        keyframes = torch.arange(0, self.num_patches, step=5, device=x.device)
        all_frames = torch.arange(0, self.num_patches, device=x.device)
        masked_frames = torch.as_tensor([token not in keyframes for token in all_frames], device=x.device).unsqueeze(0)
        return masked_frames


def vis_motion(motion, val_loader, args, title="Motion Seq", filename='gm3'):
    bs, seq = motion.shape[0], motion.shape[1]
    num_joints = 21 if motion.shape[-1] == 251 else 22
    pred_pose = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
    pred_pose_xyz = recover_from_ric(torch.from_numpy(pred_pose).float().cuda(), num_joints)
    xyz = pred_pose_xyz[:1]
    xyz = xyz.reshape(bs, seq, -1, 3)
    plot_xyz = draw_to_batch(xyz.cpu().numpy(), title,
                             [os.path.join('output', args.exp_name, f'{filename}.gif')])
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

val_loader = eval_dataloader(args.dataname, False, 32, w_vectorizer, unit_length=2 ** args.down_t)

use_mask_token = args.with_mask_token or args.with_mask_token_eval
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
                 args.vq_norm,
                 mask_token=use_mask_token,
                 mask_input=False)

if args.resume_pth:
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt_vqvae = torch.load(args.resume_pth, map_location='cpu')
    vqvae.load_state_dict(ckpt_vqvae['net'], strict=True)

vqvae.eval()
vqvae.cuda()

m3 = MMM(mask_ratio=args.mask_ratio)

if args.with_mask_token_eval:
    ckpt_gm3 = torch.load(os.path.join('output', args.exp_name, 'net_last.pth'), map_location='cpu')
    m3.load_state_dict(ckpt_gm3['net'], strict=True)
    m3.eval()
    m3.cuda()

    mask_token = vqvae.vqvae.mask_token
    args.nb_joints = 22

    val_loader = eval_dataloader(args.dataname, True, 1, w_vectorizer, unit_length=2 ** args.down_t)

    c = 0
    draw_org = []
    draw_pred = []
    for i, batch in enumerate(val_loader):
        if i>2:
            exit()
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch
        motion = motion.cuda()

        # mask input
        N, T, h = motion.shape # batch, n_token, hidden_dim
        x_in = vqvae.vqvae.preprocess(motion)

        # mask
        mask_generator = KeyFrameMask(T)
        masks = mask_generator(x_in)  # first dim should be Batch. return_size:256x64
        Ntotal_masks = masks.size(1) # return_size:64
        mask_tokens = mask_token.repeat(N, Ntotal_masks, 1).to(dtype=x_in.dtype) # return_size:256x64x263
        x_in = x_in.permute(0,2,1) # return_size:256x64x263
        mask_tokens = mask_tokens.view(-1, 263)
        masks = masks.view(-1)
        mask_tokens[~masks] = x_in.view(N * T, h)[~masks]
        x_in = mask_tokens.view(N, -1, h)
        #x_in = vqvae.vqvae.preprocess(x_in) #x_in.permute(0, 2, 1)

        # vqvae encoder
        code_idx = vqvae.encode(x_in)
        quants = vqvae.vqvae.quantizer.dequantize(code_idx)

        # masked encoder decoder
        # TODO: add conditions to use only vqvae or both vqvae and gm3
        # if not vqvae_only:
        logits, mask, target = m3(quants)
        x_d = logits.view(1, -1, vqvae.vqvae.code_dim).permute(0, 2, 1).contiguous()
        #else:
        # vqvae decoder
        #x_d = quants.view(1, -1, vqvae.vqvae.code_dim).permute(0, 2, 1).contiguous()

        x_decoder = vqvae.vqvae.decoder(x_d)
        motion_pred = vqvae.vqvae.postprocess(x_decoder)
        print(f"Seq {i}")
        print(F.mse_loss(
            motion,
            motion_pred,
            reduction="mean",
        ))
        #print(mask)
        #print()
        # vis
        vis_motion(x_in, val_loader, args, title="GT Seq", filename=f'keyframe_gt{i}')
        vis_motion(motion_pred, val_loader, args, title="Pred Seq", filename=f'keyframe_pred{i}')
    exit()


if args.eval:
    logger.info('loading checkpoint from {}'.format(args.exp_name))
    ckpt_gm3 = torch.load(os.path.join('output', args.exp_name, 'net_last.pth'), map_location='cpu')
    m3.load_state_dict(ckpt_gm3['net'], strict=True)
    m3.eval()
    m3.cuda()

    args.nb_joints = 22
    val_loader = eval_dataloader(args.dataname, True, 1, w_vectorizer, unit_length=2 ** args.down_t)
    c = 0
    draw_org = []
    draw_pred = []
    for i, batch in enumerate(val_loader):
        if not i>1:
            continue
        if i > 3:
            exit()
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch
        motion = motion.cuda()
        # vqvae encoder
        code_idx = vqvae.encode(motion)
        quants = vqvae.vqvae.quantizer.dequantize(code_idx)
        # masked encoder decoder
        logits, mask, target = m3(quants)
        # vqvae decoder
        x_d = logits.view(1, -1, vqvae.vqvae.code_dim).permute(0, 2, 1).contiguous()
        x_decoder = vqvae.vqvae.decoder(x_d)
        motion_pred = vqvae.vqvae.postprocess(x_decoder)
        print(f"Seq {i}")
        print(F.mse_loss(
            motion,
            motion_pred,
            reduction="mean",
        ))
        print(mask)
        print()
        # vis
        vis_motion(motion, val_loader, args, title="GT Seq", filename=f'99gm3_gt{i}')
        vis_motion(motion_pred, val_loader, args, title="Pred Seq", filename=f'99gm3_pred{i}')




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
print("Finished. Saving Model.")
torch.save({'net': m3.state_dict()}, os.path.join(args.out_dir, 'net_last.pth'))
print("Done.")