import csv
import glob
import os

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.transform import AffineTransform, warp

from models.decoder import TILE

AFF_DEG, AFF_SCALE = 15.0, 0.2
CONTRAST = (0.5, 1.5)
BRIGHT = (0.9, 1.1)


def ctc_pairs(root, dataset, seq, kind):
    segs = sorted(glob.glob(os.path.join(root, dataset, '%s_%s' % (seq, kind), 'SEG', '*.tif')))
    out = []
    for sp in segs:
        num = os.path.basename(sp).replace('man_seg', '').replace('.tif', '')
        ip = os.path.join(root, dataset, seq, 't%s.tif' % num)
        if os.path.exists(ip):
            out.append((ip, sp, int(num)))
    return out


def _merge_gt_st(root, dataset, seq):
    gt = {t: (ip, sp) for ip, sp, t in ctc_pairs(root, dataset, seq, 'GT')}
    st = {t: (ip, sp) for ip, sp, t in ctc_pairs(root, dataset, seq, 'ST')}
    return [(*(gt[t] if t in gt else st[t]), t, 'GT' if t in gt else 'ST') for t in sorted(set(gt) | set(st))]


def train_pairs(root, dataset):
    return _merge_gt_st(root, dataset, '01')


def test_pairs(root, dataset, which='all02'):
    if which == 'gt02':
        return [(ip, sp, t, 'GT') for ip, sp, t in ctc_pairs(root, dataset, '02', 'GT')]
    return _merge_gt_st(root, dataset, '02')


def csv_pairs(csv_path, split):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return [(r['image_path'], r['mask_path'], i, 'GT') for i, r in enumerate(rows) if r['set'] == split]


def get_pairs(ctc_root=None, dataset=None, data_csv=None, split='train', test='all02'):
    if data_csv:
        return csv_pairs(data_csv, split)
    return train_pairs(ctc_root, dataset) if split == 'train' else test_pairs(ctc_root, dataset, test)


def norm_cp(a, lo=1, hi=99):
    a = a.astype(np.float32)
    p1, p2 = np.percentile(a, [lo, hi])
    return ((a - p1) / (p2 - p1 + 1e-8)).astype(np.float32)


def gapfill(m):
    m = np.asarray(m, np.int32).copy()
    z = m == 0
    g = np.zeros_like(z)
    g[:, 1:-1] |= z[:, 1:-1] & (m[:, :-2] > 0) & (m[:, 2:] > 0) & (m[:, :-2] != m[:, 2:])
    g[1:-1, :] |= z[1:-1, :] & (m[:-2, :] > 0) & (m[2:, :] > 0) & (m[:-2, :] != m[2:, :])
    if g.any():
        m[g] = ndi.grey_dilation(m, size=3)[g]
    return m


def jitter(x, rng):
    c = rng.uniform(*CONTRAST)
    b = rng.uniform(*BRIGHT)
    mu = x.mean()
    return ((x - mu) * c + mu) * b


def pad_to(a, h, w):
    return np.pad(a, ((0, max(0, h - a.shape[0])), (0, max(0, w - a.shape[1]))), mode='reflect')


def inst_dist(lab, min_area=10):
    out = np.zeros(lab.shape, np.float32)
    for v in np.unique(lab):
        if v == 0:
            continue
        m = lab == v
        if m.sum() < min_area:
            continue
        d = ndi.distance_transform_edt(m)
        mx = d.max()
        if mx > 0:
            out[m] = d[m] / mx
    return out


def md_rule(labels, min_area=10):
    ar = [a for lb in labels for v, a in zip(*np.unique(lb[lb > 0], return_counts=True)) if a >= min_area]
    return int(round(float(np.sqrt(np.median(ar) / np.pi))))


class AugSet(torch.utils.data.Dataset):
    def __init__(self, images, labels, n, seed):
        self.images, self.labels, self.n = images, labels, n
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = self.rng
        while True:
            k = rng.integers(len(self.images))
            im, lb = self.images[k], self.labels[k]
            h, w = im.shape
            cy, cx = rng.uniform(TILE / 2, h - TILE / 2), rng.uniform(TILE / 2, w - TILE / 2)
            th = np.deg2rad(rng.uniform(-AFF_DEG, AFF_DEG))
            sc = rng.uniform(1 - AFF_SCALE, 1 + AFF_SCALE)
            T = (AffineTransform(translation=(-TILE / 2, -TILE / 2)) + AffineTransform(rotation=th, scale=1 / sc)
                 + AffineTransform(translation=(cx, cy)))
            a = warp(im, T, output_shape=(TILE, TILE), order=1, mode='constant', cval=0, preserve_range=True)
            b = warp(lb, T, output_shape=(TILE, TILE), order=0, mode='constant', cval=0,
                     preserve_range=True).astype(np.int32)
            if rng.random() < .5:
                a, b = a[:, ::-1], b[:, ::-1]
            if rng.random() < .5:
                a, b = a[::-1], b[::-1]
            if (b > 0).sum() < 50:
                continue
            a = jitter(np.ascontiguousarray(a, np.float32), rng)
            b = np.ascontiguousarray(b)
            return (torch.from_numpy(a.astype(np.float32)),
                    torch.stack([torch.from_numpy((b > 0).astype(np.float32)), torch.from_numpy(inst_dist(b))]))


def worker_init(_):
    ds = torch.utils.data.get_worker_info().dataset
    ds.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)
