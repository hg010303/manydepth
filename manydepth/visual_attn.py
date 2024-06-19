# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil
import torchvision

from .utils import readlines
from .options import MonodepthOptions
from manydepth import datasets, networks
from .layers import transformation_from_parameters, disp_to_depth
import tqdm

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = "splits"

STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    frames_to_load = [0]
    if opt.use_future_frame:
        frames_to_load.append(1)
    for idx in range(-1, -1 - opt.num_matching_frames, -1):
        if idx not in frames_to_load:
            frames_to_load.append(idx)

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        
        from manydepth.networks.croco.croco_downstream import CroCoDownstreamBinocular, croco_args_from_ckpt
        from manydepth.networks.croco.pos_embed import interpolate_pos_embed
        from manydepth.networks.croco.head_downstream import PixelwiseTaskWithDPT

        ckpt = torch.load(opt.croco_pretrain_path, 'cpu')
        croco_args = croco_args_from_ckpt(ckpt)
        croco_args['img_size'] = (192, 640)
        croco_args['is_eval'] = True
        HEIGHT, WIDTH = 192, 640

        head = PixelwiseTaskWithDPT()
        head.num_channels = 1
        # build model and load pretrained weights
        model = CroCoDownstreamBinocular(head, **croco_args)
        interpolate_pos_embed(model, ckpt['model'])
        msg = model.load_state_dict(ckpt['model'], strict=False)
        print(msg)

        depth_dict = torch.load(os.path.join(opt.load_weights_folder,"depth_network.pth"))
        model.load_state_dict(depth_dict)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()


        pose_enc_dict = torch.load(os.path.join(opt.load_weights_folder, "pose_encoder.pth"))
        pose_dec_dict = torch.load(os.path.join(opt.load_weights_folder, "pose.pth"))

        pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
        pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
                                        num_frames_to_predict_for=2)

        pose_enc.load_state_dict(pose_enc_dict, strict=True)
        pose_dec.load_state_dict(pose_dec_dict, strict=True)

        if torch.cuda.is_available():
            pose_enc.cuda()
            pose_dec.cuda()

        pose_enc.eval()
        pose_dec.eval()


        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                        HEIGHT, WIDTH,
                                        frames_to_load, 4,
                                        is_train=False)
                                            
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                    pin_memory=True, drop_last=False)

        pred_disps = []
        gt_depths = []

        print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))

        resize = torchvision.transforms.Resize((192, 640))

        # do inference
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(dataloader)):
                input_color = data[('color', 0, 0)]
                gt_depth = data['depth_gt']
                if torch.cuda.is_available():
                    input_color = input_color.cuda()

                if opt.eval_teacher:
                    output = encoder(input_color)
                    output = depth_decoder(output)
                else:

                    if opt.static_camera:
                        for f_i in frames_to_load:
                            data["color", f_i, 0] = data[('color', 0, 0)]

                    # predict poses
                    pose_feats = {f_i: data["color", f_i, 0] for f_i in frames_to_load}
                    if torch.cuda.is_available():
                        pose_feats = {k: v.cuda() for k, v in pose_feats.items()}
                    # compute pose from 0->-1, -1->-2, -2->-3 etc and multiply to find 0->-3
                    for fi in frames_to_load[1:]:
                        if fi < 0:
                            pose_inputs = [pose_feats[fi], pose_feats[fi + 1]]
                            pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                            axisangle, translation = pose_dec(pose_inputs)
                            pose = transformation_from_parameters(
                                axisangle[:, 0], translation[:, 0], invert=True)

                            # now find 0->fi pose
                            if fi != -1:
                                pose = torch.matmul(pose, data[('relative_pose', fi + 1)])

                        else:
                            pose_inputs = [pose_feats[fi - 1], pose_feats[fi]]
                            pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
                            axisangle, translation = pose_dec(pose_inputs)
                            pose = transformation_from_parameters(
                                axisangle[:, 0], translation[:, 0], invert=False)

                            # now find 0->fi pose
                            if fi != 1:
                                pose = torch.matmul(pose, data[('relative_pose', fi - 1)])

                        data[('relative_pose', fi)] = pose

                    lookup_frames = [data[('color', idx, 0)] for idx in frames_to_load[1:]]
                    lookup_frames = torch.stack(lookup_frames, 1)  # batch x frames x 3 x h x w

                    relative_poses = [data[('relative_pose', idx)] for idx in frames_to_load[1:]]
                    relative_poses = torch.stack(relative_poses, 1)

                    K = data[('K', 2)]  # quarter resolution for matching
                    invK = data[('inv_K', 2)]

                    if torch.cuda.is_available():
                        lookup_frames = lookup_frames.cuda()
                        relative_poses = relative_poses.cuda()
                        K = K.cuda()
                        invK = invK.cuda()

                    output = {}
                    out_depth, _ = model(input_color, lookup_frames.squeeze(dim=1))
                    
                    ###### vis attn map
                    attention_vis_path = './attention_vis'
                    attn_maps = [model.dec_blocks[i].cross_attn.attn_map for i in range(12)]

                    img = input_color[0].clone()
                    prev_img = lookup_frames[:, 0].clone()

                    vis_depth = out_depth[0].clone()
                    vis_depth = vis_depth[0].cpu().detach().numpy()

                    vmax = np.percentile(vis_depth, 95)
                    normalizer = mpl.colors.Normalize(vmin=vis_depth.min(), vmax=vmax)
                    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                    colormapped_im = (mapper.to_rgba(vis_depth[:,:])[:, :, :3] * 255).astype(np.uint8)
                    vis_depth = pil.fromarray(colormapped_im)
                    vis_depth.save(f'./{attention_vis_path}/{i}_depth.png')

                    # height, width = 6,20
                    height = torch.randint(0, 12, (1,)).item()
                    width = torch.randint(0, 40, (1,)).item()

                    img[...,height*16:(height+1)*16,width*16:(width+1)*16] = torch.tensor([1.0,0.0,0.0]).unsqueeze(dim=1).unsqueeze(dim=1).expand(3,16,16)
                    img_tmp = img.clone().permute(1,2,0).cpu().detach().numpy()
                    img_input = pil.fromarray((img_tmp*255).astype(np.uint8))
                    img_input.save(f'./{attention_vis_path}/{i}_img_input.png')

                    for j in range(len(attn_maps[0])):
                        attn_map = attn_maps[0][j].mean(dim=0)
                        attn_map = attn_map.reshape(12,40,-1)
                        attn_map = attn_map[height][width].reshape(12,40)

                        img_tmp = lookup_frames[0].squeeze(dim=0).clone().permute(1,2,0).cpu().detach().numpy()

                        attn_map = resize(attn_map.unsqueeze(0).unsqueeze(0)).squeeze(0).permute(1,2,0).cpu().detach().numpy()

                        vmax = np.percentile(attn_map, 100)
                        normalizer = mpl.colors.Normalize(vmin=attn_map.min(), vmax=vmax)
                        mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')
                        colormapped_im = (mapper.to_rgba(attn_map[:,:,0])[:, :, :3] * 255).astype(np.uint8)
                        attn_map = cv2.addWeighted((img_tmp*255).astype(np.uint8), 0.6, colormapped_im, 0.4, 0)
                        attn_map = pil.fromarray(attn_map)
                        attn_map.save(f'./{attention_vis_path}/{i}_{j}th_attnmap.png')

                    ###########################



                    output[('disp',0)] = out_depth


                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)
                gt_depths.append(gt_depth)

        pred_disps = np.concatenate(pred_disps)
        gt_depths = np.concatenate(gt_depths)

        print('finished predicting!')


    if opt.save_pred_disps:

        tag = "multi"
        output_path = os.path.join(
            opt.load_weights_folder, "{}_{}_split.npy".format(tag, opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)


    gt_depths = gt_depths.squeeze()

    print("-> Evaluating")

    errors = []
    ratios = []
    for i in tqdm.tqdm(range(pred_disps.shape[0])):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = np.squeeze(pred_disps[i])
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if opt.save_pred_disps:
        print("saving errors")
        if opt.zero_cost_volume:
            tag = "mono"
        else:
            tag = "multi"
        output_path = os.path.join(
            opt.load_weights_folder, "{}_{}_errors.npy".format(tag, opt.eval_split))
        np.save(output_path, np.array(errors))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel",
                                           "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
