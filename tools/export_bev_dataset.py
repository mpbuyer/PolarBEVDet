import argparse
import importlib
import json
import os
import os.path as osp
import re

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export BEV feature images + 3D boxes as a small dataset')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument(
        '--split',
        choices=['val', 'test'],
        default='val',
        help='dataset split from config.data')
    parser.add_argument(
        '--out-dir',
        default='/kaggle/working/polarbevdet_bev_dataset',
        help='output directory')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='export at most this many samples')
    parser.add_argument(
        '--workers-per-gpu',
        type=int,
        default=2,
        help='dataloader workers')
    parser.add_argument(
        '--samples-per-gpu',
        type=int,
        default=1,
        help='batch size per gpu')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.1,
        help='minimum box score to keep in exported box json')
    parser.add_argument(
        '--feature-space',
        choices=['cartesian', 'polar'],
        default='cartesian',
        help='save PCA BEV image in cartesian space or native polar space')
    parser.add_argument(
        '--bev-source',
        choices=['det_class_rgb', 'det_heatmap', 'pre_encoder', 'post_encoder'],
        default='det_class_rgb',
        help='which BEV tensor to visualize')
    parser.add_argument(
        '--cart-size',
        type=int,
        default=512,
        help='output size for cartesian BEV image (square)')
    parser.add_argument(
        '--use-fp16-wrap',
        action='store_true',
        help='use mmcv fp16 wrapper for inference (disabled by default)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options, key=value')
    return parser.parse_args()


def import_plugin(cfg, config_path):
    if not hasattr(cfg, 'plugin') or not cfg.plugin:
        return

    if hasattr(cfg, 'plugin_dir'):
        module_dir = os.path.dirname(cfg.plugin_dir)
    else:
        module_dir = os.path.dirname(config_path)

    module_dir = module_dir.split('/')
    module_path = module_dir[0]
    for item in module_dir[1:]:
        module_path = module_path + '.' + item
    importlib.import_module(module_path)


def to_python_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sanitize_for_filename(value):
    value = str(value)
    value = re.sub(r'[^A-Za-z0-9._-]+', '-', value)
    return value.strip('-') or 'unknown'


def unwrap_meta_obj(obj):
    while hasattr(obj, 'data'):
        obj = obj.data
    if isinstance(obj, (list, tuple)) and len(obj) == 1:
        return unwrap_meta_obj(obj[0])
    return obj


def get_img_metas_from_data(data):
    img_metas = unwrap_meta_obj(data['img_metas'])
    if isinstance(img_metas, dict):
        return [img_metas]
    if isinstance(img_metas, (list, tuple)):
        out = []
        for item in img_metas:
            item = unwrap_meta_obj(item)
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, (list, tuple)):
                for sub_item in item:
                    sub_item = unwrap_meta_obj(sub_item)
                    if isinstance(sub_item, dict):
                        out.append(sub_item)
        if len(out) > 0:
            return out
    return [{}]


def pca_feature_to_rgb(bev_feature):
    # bev_feature: (C, H, W)
    feat = bev_feature.detach().float().cpu()
    c, h, w = feat.shape
    pixels = feat.permute(1, 2, 0).reshape(-1, c)
    mean = pixels.mean(dim=0, keepdim=True)
    std = pixels.std(dim=0, keepdim=True).clamp(min=1e-6)
    pixels = (pixels - mean) / std

    q = min(3, c, pixels.shape[0])
    if q == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    _, s, v = torch.pca_lowrank(pixels, q=q, center=False, niter=4)
    reduced = pixels @ v[:, :q]
    if q < 3:
        reduced = F.pad(reduced, (0, 3 - q))

    img = reduced.reshape(h, w, 3).permute(2, 0, 1)
    norm_channels = []
    for ch in img:
        lo = torch.quantile(ch, 0.01)
        hi = torch.quantile(ch, 0.99)
        denom = (hi - lo).clamp(min=1e-6)
        norm = ((ch - lo) / denom).clamp(0, 1)
        norm_channels.append(norm)
    img = torch.stack(norm_channels, dim=0)
    img = (img * 255.0).byte().permute(1, 2, 0).numpy()

    # If PCA visualization collapses to near-uniform color, fall back to
    # highest-variance channels, then gradient-energy as last resort.
    sv_energy = float((s[0] / (s.sum() + 1e-6)).item()) if s.numel() > 0 else 1.0
    if img.std() < 3.0 or sv_energy > 0.995:
        spatial_std = feat.flatten(1).std(dim=1)
        topk = torch.topk(spatial_std, k=min(3, feat.shape[0])).indices
        ch_img = feat[topk]
        if ch_img.shape[0] < 3:
            ch_img = F.pad(ch_img, (0, 0, 0, 0, 0, 3 - ch_img.shape[0]))
        ch_norm = []
        for ch in ch_img:
            lo = torch.quantile(ch, 0.01)
            hi = torch.quantile(ch, 0.99)
            ch_norm.append(((ch - lo) / (hi - lo + 1e-6)).clamp(0, 1))
        img = (torch.stack(ch_norm, dim=0) * 255.0).byte().permute(1, 2, 0).numpy()

    if img.std() < 3.0:
        mag = torch.norm(feat, dim=0)
        gy = torch.zeros_like(mag)
        gx = torch.zeros_like(mag)
        gy[1:, :] = torch.abs(mag[1:, :] - mag[:-1, :])
        gx[:, 1:] = torch.abs(mag[:, 1:] - mag[:, :-1])
        edge = gx + gy
        lo = torch.quantile(edge, 0.01)
        hi = torch.quantile(edge, 0.995)
        edge = ((edge - lo) / (hi - lo + 1e-6)).clamp(0, 1)
        edge = torch.pow(edge, 0.7)
        edge_u8 = (edge * 255.0).byte().numpy()
        img = np.stack([edge_u8, edge_u8, edge_u8], axis=-1)

    return img


def stretch_rgb_percentile(rgb_image, low=1.0, high=99.0, valid_mask=None):
    img = rgb_image.astype(np.float32)
    out = np.zeros_like(img, dtype=np.float32)
    for ch in range(3):
        data = img[..., ch]
        vals = data if valid_mask is None else data[valid_mask]
        if vals.size == 0:
            continue
        lo = np.percentile(vals, low)
        hi = np.percentile(vals, high)
        denom = max(hi - lo, 1e-6)
        norm = np.clip((data - lo) / denom, 0.0, 1.0)
        out[..., ch] = norm
    return (out * 255.0).astype(np.uint8)


def heatmap_to_rgb(heatmap_tensor):
    hm = heatmap_tensor.detach().float().cpu()
    if hm.ndim == 3 and hm.shape[0] == 1:
        hm = hm.squeeze(0)
    if hm.ndim != 2:
        raise RuntimeError(f'Expected heatmap shape (H, W), got {tuple(hm.shape)}')
    lo = torch.quantile(hm, 0.01)
    hi = torch.quantile(hm, 0.995)
    hm = ((hm - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    hm_u8 = (hm * 255.0).byte().numpy()
    # cv2 color map returns BGR
    hm_bgr = cv2.applyColorMap(hm_u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB)


def class_heatmap_to_rgb(class_heatmap_tensor):
    hm = class_heatmap_tensor.detach().float().cpu()  # (C, H, W)
    if hm.ndim != 3:
        raise RuntimeError(f'Expected class heatmap shape (C, H, W), got {tuple(hm.shape)}')

    c, h, w = hm.shape
    top2 = torch.topk(hm, k=min(2, c), dim=0)
    top1_prob = top2.values[0]  # (H, W)
    if c > 1:
        margin = top2.values[0] - top2.values[1]
    else:
        margin = top1_prob
    cls_idx = top2.indices[0]  # (H, W)

    palette = torch.tensor([
        [244, 67, 54], [33, 150, 243], [76, 175, 80], [255, 193, 7], [156, 39, 176],
        [0, 188, 212], [255, 87, 34], [139, 195, 74], [63, 81, 181], [255, 152, 0],
        [121, 85, 72], [96, 125, 139], [233, 30, 99], [3, 169, 244], [205, 220, 57],
        [255, 235, 59], [0, 150, 136], [103, 58, 183], [255, 111, 0], [0, 121, 107],
    ], dtype=torch.float32)
    palette = palette / 255.0

    # if more classes than palette, wrap around deterministically
    cls_mod = torch.remainder(cls_idx, palette.shape[0])
    base_rgb = palette[cls_mod.long()]  # (H, W, 3)

    # local contrast enhancement over confidence
    prob_np = top1_prob.numpy()
    prob_blur = cv2.GaussianBlur(prob_np, (0, 0), sigmaX=2.0, sigmaY=2.0)
    local = np.abs(prob_np - prob_blur)
    lo = np.percentile(local, 1.0)
    hi = np.percentile(local, 99.5)
    local = np.clip((local - lo) / (max(hi - lo, 1e-6)), 0.0, 1.0)

    mar = margin
    mar_lo = torch.quantile(mar, 0.01)
    mar_hi = torch.quantile(mar, 0.995)
    mar = ((mar - mar_lo) / (mar_hi - mar_lo + 1e-6)).clamp(0, 1).numpy()

    # value map combines confidence and local contrast
    val = np.clip(0.15 + 0.45 * mar + 0.60 * local, 0.0, 1.0)
    rgb = base_rgb.numpy() * val[..., None]
    rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    if rgb.std() < 3.0:
        # fallback to confidence heatmap if class map is still too uniform
        return heatmap_to_rgb(top1_prob)
    return rgb


def polar_rgb_to_cartesian(rgb_image, azimuth_cfg, radius_cfg, cart_size):
    # rgb_image: np.ndarray(H, W, 3) in native polar BEV layout.
    img = torch.from_numpy(rgb_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    _, _, h, w = img.shape

    az_min, az_max, az_step = azimuth_cfg
    r_min, r_max, r_step = radius_cfg

    x_vals = torch.linspace(r_max, -r_max, cart_size)
    y_vals = torch.linspace(r_max, -r_max, cart_size)
    x_grid = x_vals[:, None].expand(cart_size, cart_size)
    y_grid = y_vals[None, :].expand(cart_size, cart_size)

    radius = torch.sqrt(x_grid**2 + y_grid**2)
    azimuth = torch.atan2(y_grid, x_grid)

    u = (azimuth - az_min) / az_step
    u = torch.clamp(u, 0, w - 1 - 1e-4)
    v = (radius - r_min) / r_step

    grid_x = (u / (w - 1)) * 2 - 1
    grid_y = (v / (h - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    sampled = F.grid_sample(
        img,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True)
    sampled = sampled.squeeze(0)

    outside = (radius < r_min) | (radius > r_max)
    sampled[:, outside] = 0

    sampled = (sampled.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).numpy()
    valid_mask = (~outside).cpu().numpy()
    sampled = stretch_rgb_percentile(sampled, low=1.0, high=99.0, valid_mask=valid_mask)
    sampled[~valid_mask] = 0
    return sampled


def build_token_info_map(dataset):
    token_map = {}
    data_infos = getattr(dataset, 'data_infos', [])
    for info in data_infos:
        token = info.get('token', info.get('sample_idx'))
        if token is None:
            continue
        token_map[str(token)] = info
    return token_map


def build_output_stem(scene_name, scene_token, frame_idx, sample_token):
    scene_name = scene_name if scene_name is not None else scene_token
    scene_name = sanitize_for_filename(scene_name if scene_name is not None else 'scene')
    frame_repr = 'unknown'
    if frame_idx is not None:
        try:
            frame_repr = f'{int(frame_idx):04d}'
        except (TypeError, ValueError):
            frame_repr = sanitize_for_filename(frame_idx)
    token_repr = sanitize_for_filename(sample_token if sample_token is not None else 'sample')
    return f'{scene_name}__frame-{frame_repr}__token-{token_repr}'


def export_box_json(path, pts_bbox, classes, score_thr):
    boxes = pts_bbox['boxes_3d'].tensor.detach().cpu().numpy()
    scores = pts_bbox['scores_3d'].detach().cpu().numpy()
    labels = pts_bbox['labels_3d'].detach().cpu().numpy().astype(np.int64)

    keep = scores >= score_thr
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    label_names = [classes[i] if 0 <= i < len(classes) else str(i) for i in labels.tolist()]

    payload = {
        'boxes_3d': boxes.tolist(),
        'scores_3d': scores.tolist(),
        'labels_3d': labels.tolist(),
        'label_names': label_names,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    return int(keep.sum())


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    import_plugin(cfg, args.config)

    if args.samples_per_gpu > 1:
        dataset_cfg = cfg.data[args.split]
        if isinstance(dataset_cfg, dict):
            dataset_cfg.pipeline = replace_ImageToTensor(dataset_cfg.pipeline)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    dataset_cfg = cfg.data[args.split]
    if isinstance(dataset_cfg, dict):
        dataset_cfg.test_mode = True
    elif isinstance(dataset_cfg, list):
        for one_cfg in dataset_cfg:
            one_cfg.test_mode = True
    else:
        raise TypeError(f'Unsupported cfg.data.{args.split} type: {type(dataset_cfg)}')

    dataset = build_dataset(dataset_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.get('nonshuffler_sampler', None),
    )

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None and args.use_fp16_wrap:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this exporter (MMDataParallel path).')

    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    mmcv.mkdir_or_exist(args.out_dir)
    bev_dir = osp.join(args.out_dir, 'bev_images')
    box_dir = osp.join(args.out_dir, 'boxes')
    mmcv.mkdir_or_exist(bev_dir)
    mmcv.mkdir_or_exist(box_dir)

    manifest_path = osp.join(args.out_dir, 'manifest.jsonl')
    summary_path = osp.join(args.out_dir, 'summary.json')

    token_info_map = build_token_info_map(dataset)
    total_target = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    prog_bar = mmcv.ProgressBar(total_target)

    processed = 0
    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        for data in data_loader:
            with torch.no_grad():
                results = model(
                    return_loss=False,
                    rescale=True,
                    return_bev_features=True,
                    **data)

            if not isinstance(results, list):
                raise RuntimeError('Expected model output to be a list per batch.')

            img_metas = get_img_metas_from_data(data)
            if not isinstance(img_metas, (list, tuple)):
                img_metas = [img_metas]

            for batch_idx, pred in enumerate(results):
                if args.max_samples is not None and processed >= args.max_samples:
                    break

                img_meta = img_metas[batch_idx] if batch_idx < len(img_metas) else {}
                if not isinstance(img_meta, dict):
                    img_meta = {}
                sample_token = to_python_scalar(pred.get('sample_idx', img_meta.get('sample_idx')))
                scene_token = to_python_scalar(pred.get('scene_token', img_meta.get('scene_token')))
                scene_name = to_python_scalar(pred.get('scene_name', img_meta.get('scene_name')))
                frame_idx = to_python_scalar(pred.get('frame_idx', img_meta.get('frame_idx')))
                timestamp = to_python_scalar(pred.get('timestamp', img_meta.get('timestamp')))

                info = token_info_map.get(str(sample_token), {})
                if scene_token is None:
                    scene_token = info.get('scene_token')
                if scene_name is None:
                    scene_name = info.get('scene_name')
                if frame_idx is None:
                    frame_idx = info.get('frame_idx', processed)
                if timestamp is None:
                    timestamp = info.get('timestamp')

                stem = build_output_stem(scene_name, scene_token, frame_idx, sample_token)
                bev_rel = osp.join('bev_images', f'{stem}.png')
                box_rel = osp.join('boxes', f'{stem}.json')
                bev_path = osp.join(args.out_dir, bev_rel)
                box_path = osp.join(args.out_dir, box_rel)

                if args.bev_source == 'det_class_rgb':
                    bev_feature = pred.get('bev_det_cls_heatmap', None)
                    is_class_heatmap = True
                    is_heatmap = False
                    if bev_feature is None:
                        bev_feature = pred.get('bev_det_heatmap', None)
                        is_class_heatmap = False
                        is_heatmap = True
                elif args.bev_source == 'det_heatmap':
                    bev_feature = pred.get('bev_det_heatmap', None)
                    is_class_heatmap = False
                    is_heatmap = True
                    if bev_feature is None:
                        bev_feature = pred.get('bev_features_pre_encoder', None)
                        is_class_heatmap = False
                        is_heatmap = False
                elif args.bev_source == 'pre_encoder':
                    bev_feature = pred.get('bev_features_pre_encoder', None)
                    if bev_feature is None:
                        bev_feature = pred.get('bev_features', None)
                    is_class_heatmap = False
                    is_heatmap = False
                else:
                    bev_feature = pred.get('bev_features', None)
                    is_class_heatmap = False
                    is_heatmap = False
                if bev_feature is None:
                    raise RuntimeError(
                        'Model output does not include bev_features. '
                        'Ensure the detector supports return_bev_features=True.')
                if processed == 0:
                    feat_stats = bev_feature.detach().float().cpu()
                    print(
                        '[Debug] bev tensor stats: '
                        f'source={args.bev_source}, '
                        f'shape={tuple(feat_stats.shape)}, '
                        f'mean={feat_stats.mean().item():.5f}, '
                        f'std={feat_stats.std().item():.5f}, '
                        f'min={feat_stats.min().item():.5f}, '
                        f'max={feat_stats.max().item():.5f}'
                    )

                if is_class_heatmap:
                    bev_img = class_heatmap_to_rgb(bev_feature)
                elif is_heatmap:
                    bev_img = heatmap_to_rgb(bev_feature)
                else:
                    if bev_feature.ndim == 4 and bev_feature.shape[0] == 1:
                        bev_feature = bev_feature.squeeze(0)
                    if bev_feature.ndim != 3:
                        raise RuntimeError(f'Unexpected bev_features shape: {tuple(bev_feature.shape)}')
                    bev_img = pca_feature_to_rgb(bev_feature)
                if args.feature_space == 'cartesian':
                    if hasattr(cfg, 'grid_config'):
                        azimuth_cfg = cfg.grid_config['azimuth']
                        radius_cfg = cfg.grid_config['radius']
                    else:
                        azimuth_cfg = cfg.model.img_view_transformer.grid_config['azimuth']
                        radius_cfg = cfg.model.img_view_transformer.grid_config['radius']
                    bev_img = polar_rgb_to_cartesian(
                        bev_img, azimuth_cfg, radius_cfg, args.cart_size)

                # OpenCV expects BGR byte order for standard display conventions.
                cv2.imwrite(bev_path, cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR))

                if 'pts_bbox' not in pred:
                    raise RuntimeError('Prediction output does not contain pts_bbox.')
                num_boxes = export_box_json(box_path, pred['pts_bbox'], dataset.CLASSES, args.score_thr)

                record = {
                    'dataset_index': processed,
                    'sample_token': sample_token,
                    'scene_token': scene_token,
                    'scene_name': scene_name,
                    'frame_idx': frame_idx,
                    'timestamp': timestamp,
                    'bev_image': bev_rel,
                    'boxes_json': box_rel,
                    'num_boxes': num_boxes,
                    'feature_space': args.feature_space,
                    'bev_source': args.bev_source,
                }
                manifest_file.write(json.dumps(record) + '\n')

                processed += 1
                prog_bar.update()

            if args.max_samples is not None and processed >= args.max_samples:
                break

    summary = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'split': args.split,
        'out_dir': args.out_dir,
        'num_samples': processed,
        'feature_space': args.feature_space,
        'bev_source': args.bev_source,
        'cart_size': args.cart_size if args.feature_space == 'cartesian' else None,
        'score_thr': args.score_thr,
        'manifest': manifest_path,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'\nExport complete: {processed} samples')
    print(f'Manifest: {manifest_path}')
    print(f'Summary:  {summary_path}')


if __name__ == '__main__':
    main()
