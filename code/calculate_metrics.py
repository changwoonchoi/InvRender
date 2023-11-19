# calculate PSNR, SSIM, LPIPS for albedo, normal, novel view synthesis images

import os
import argparse

import torch
import numpy as np
import imageio

# from render import util
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def _srgb_to_rgb(srgb):
    rgb = np.power(srgb ,2.2)
    return rgb

def _load_img(img_path, permute=True):
    img = imageio.imread(img_path)
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
    else:
        img = torch.tensor(img, dtype=torch.float32)
    if img.shape[-1] == 4:
        # white bkgd
        img[..., :3] = img[..., :3] * img[..., 3][..., None] + (1. - img[..., 3])[..., None]
    if permute:
        img = img.permute(2, 0, 1)
    return img

def _compute_rgb_scale(alpha_thres=0.9, first_gt_albedo=None, first_pred_albedo=None):
    """Computes RGB scales that match predicted albedo to ground truth,
    using just the first validation view.
    """
    pred = _srgb_to_rgb(first_pred_albedo)
    alpha = first_gt_albedo[:, :, 3]
    gt = first_gt_albedo[:, :, :3]

    # Compute color correction scales, in the linear space
    is_fg = alpha > alpha_thres
    opt_scale = []
    for i in range(3):
        x_hat = pred[:, :, i][is_fg]
        x = gt[:, :, i][is_fg]
        scale = torch.dot(x_hat, x) / torch.dot(x_hat, x_hat)
        opt_scale.append(scale.item())
    opt_scale = torch.Tensor(opt_scale)

    return opt_scale



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default=None, choices=["armadillo", "ficus", "hotdog", "lego"])
    parser.add_argument("--gt_dir", type=str, default="../data/tensoir")
    parser.add_argument("--result_dir", type=str, default="../relits")
    FLAGS = parser.parse_args()

    psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to('cuda')
    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0)).to('cuda')
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to('cuda')  # default: alexnet

    # 1) albedo
    first_gt_albedo = _load_img(os.path.join(FLAGS.gt_dir, FLAGS.scene, f"test_{0:03d}", "albedo.png"), permute=False)
    first_pred_albedo= _load_img(os.path.join(FLAGS.result_dir, f"Mat-tensoir_{FLAGS.scene}", f"{FLAGS.scene}_origin", f"albedo_0.png"), permute=False)

    opt_scale = _compute_rgb_scale(alpha_thres=0.9, first_gt_albedo=first_gt_albedo , first_pred_albedo=first_pred_albedo)

    albedo_metrics = {
        "all_psnr": [],
        "all_ssim": [],
        "all_lpips": []
    }
    all_pred_albedos = []
    all_gt_albedos = []
    for i in range(200):
        gt_albedo= _load_img(os.path.join(FLAGS.gt_dir, FLAGS.scene, f"test_{i:03d}", "albedo.png"), permute=False)
        mask = gt_albedo[:, :, 3] > 0.9
        pred_albedo= _load_img(os.path.join(FLAGS.result_dir, f"Mat-tensoir_{FLAGS.scene}", f"{FLAGS.scene}_origin", f"albedo_{i}.png"), permute=False)
        pred_albedo = _srgb_to_rgb(pred_albedo)
        pred_albedo[mask] = pred_albedo[mask] * opt_scale
        gt_albedo = gt_albedo[..., :3]
        pred_albedo = torch.clamp(pred_albedo, 0, 1)

        gt_albedo = gt_albedo.permute(2, 0, 1)
        pred_albedo = pred_albedo.permute(2, 0, 1)
        all_gt_albedos.append(gt_albedo)
        all_pred_albedos.append(pred_albedo)
    all_pred_albedos = torch.stack(all_pred_albedos, dim=0).to('cuda')
    all_gt_albedos = torch.stack(all_gt_albedos, dim=0).to('cuda')

    with torch.no_grad():
        i = 0
        chunk = 50
        while i < 200:
            albedo_metrics["all_ssim"].append(ssim(all_pred_albedos[i:i + chunk], all_gt_albedos[i:i + chunk]).item())
            albedo_metrics["all_lpips"].append(lpips(all_pred_albedos[i:i + chunk], all_gt_albedos[i:i + chunk]).item())
            i += chunk
        i = 0
        chunk = 1
        while i < 200:
            albedo_metrics["all_psnr"].append(psnr(all_pred_albedos[i:i + chunk], all_gt_albedos[i:i + chunk]).item())
            i += chunk
    torch.cuda.empty_cache()

    # 2) normal
    normal_metrics = {
        "all_psnr": [],
        "all_ssim": [],
        "all_lpips": []
    }
    all_pred_normals = []
    all_gt_normals =[]
    for i in range(200):
        gt_normal = _load_img(os.path.join(FLAGS.gt_dir, FLAGS.scene, f"test_{i:03d}", "normal.png"))[:3]
        pred_normal = _load_img(os.path.join(FLAGS.result_dir, f"Mat-tensoir_{FLAGS.scene}", f"{FLAGS.scene}_origin", f"normal_{i}.png"))
        all_gt_normals.append(gt_normal)
        all_pred_normals.append(pred_normal)
    all_gt_normals = torch.stack(all_gt_normals, dim=0).to('cuda')
    all_pred_normals = torch.stack(all_pred_normals, dim=0).to('cuda')

    with torch.no_grad():
        i = 0
        chunk = 50
        while i < 200:
            normal_metrics["all_ssim"].append(ssim(all_pred_normals[i:i + chunk], all_gt_normals[i:i + chunk]).item())
            normal_metrics["all_lpips"].append(lpips(all_pred_normals[i:i + chunk], all_gt_normals[i:i + chunk]).item())
            i += chunk
        i = 0
        chunk = 1
        while i < 200:
            normal_metrics["all_psnr"].append(psnr(all_pred_normals[i:i + chunk], all_gt_normals[i:i + chunk]).item())
            i += chunk
    torch.cuda.empty_cache()

    # 3) novel view synthesis
    img_metrics = {
        "all_psnr": [],
        "all_ssim": [],
        "all_lpips": []
    }
    all_pred_imgs = []
    all_gt_imgs = []
    for i in range(200):
        gt_img= _load_img(os.path.join(FLAGS.gt_dir, FLAGS.scene, f"test_{i:03d}", "rgba.png"))[:3]
        pred_img = _load_img(os.path.join(FLAGS.result_dir, f"Mat-tensoir_{FLAGS.scene}", f"{FLAGS.scene}_origin", f"sg_rgb_bg_{i}.png"))
        all_pred_imgs.append(pred_img)
        all_gt_imgs.append(gt_img)
    all_pred_imgs = torch.stack(all_pred_imgs, dim=0).to('cuda')
    all_gt_imgs = torch.stack(all_gt_imgs, dim=0).to('cuda')

    with torch.no_grad():
        i = 0
        chunk = 50
        while i < 200:
            img_metrics["all_ssim"].append(ssim(all_pred_imgs[i:i + chunk], all_gt_imgs[i:i + chunk]).item())
            img_metrics["all_lpips"].append(lpips(all_pred_imgs[i:i + chunk], all_gt_imgs[i:i + chunk]).item())
            i += chunk
        i = 0
        chunk = 1
        while i < 200:
            img_metrics["all_psnr"].append(psnr(all_pred_imgs[i:i + chunk], all_gt_imgs[i:i + chunk]).item())
            i += chunk
    torch.cuda.empty_cache()

    # Summary
    print("========================================================")
    print("Albedo")
    print("PSNR: ", np.mean(albedo_metrics["all_psnr"]))
    print("SSIM: ", np.mean(albedo_metrics["all_ssim"]))
    print("LPIPS: ", np.mean(albedo_metrics["all_lpips"]))
    print("========================================================")
    print("Normal")
    print("PSNR: ", np.mean(normal_metrics["all_psnr"]))
    print("SSIM: ", np.mean(normal_metrics["all_ssim"]))
    print("LPIPS: ", np.mean(normal_metrics["all_lpips"]))
    print("========================================================")
    print("Novel View")
    print("PSNR: ", np.mean(img_metrics["all_psnr"]))
    print("SSIM: ", np.mean(img_metrics["all_ssim"]))
    print("LPIPS: ", np.mean(img_metrics["all_lpips"]))
    print("========================================================")
