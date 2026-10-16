import collections

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from dataset import norm_cp, pad_to
from models.decoder import TILE

STRIDE = 64


def tiles(h, w, s=TILE, st=STRIDE):
    ys = list(range(0, max(h - s, 0) + 1, st)) or [0]
    xs = list(range(0, max(w - s, 0) + 1, st)) or [0]
    if ys[-1] + s < h:
        ys.append(h - s)
    if xs[-1] + s < w:
        xs.append(w - s)
    return [(y, x) for y in ys for x in xs]


@torch.no_grad()
def predict_fields(img, encoder, head):
    head.eval()
    g = norm_cp(pad_to(img, TILE, TILE))
    gh, gw = g.shape
    acc = np.zeros((2, gh, gw), np.float32)
    cnt = np.zeros((gh, gw), np.float32)
    for y, x in tiles(gh, gw):
        t = torch.from_numpy(np.ascontiguousarray(g[y:y + TILE, x:x + TILE], np.float32))[None]
        _, fu, dh = head(encoder.tokens(t))
        acc[:, y:y + TILE, x:x + TILE] += torch.stack([torch.sigmoid(fu[0]), dh[0]]).float().cpu().numpy()
        cnt[y:y + TILE, x:x + TILE] += 1
    return (acc / np.maximum(cnt, 1))[:, :img.shape[0], :img.shape[1]]


def decode(fields, kind='hmax', value=0.25, thr=0.5):
    prob, dist = fields
    fg = prob > thr
    if not fg.any():
        return np.zeros(fg.shape, np.int32)
    f = dist * fg
    if kind == 'md':
        markers = np.zeros(fg.shape, np.int32)
        for i, (yy, xx) in enumerate(peak_local_max(f, min_distance=int(value), labels=fg, exclude_border=False), 1):
            markers[yy, xx] = i
    elif kind == 'hmax':
        markers, _ = ndi.label(h_maxima(f, value) & fg)
    else:
        raise ValueError(kind)
    return watershed(-f, markers, mask=fg).astype(np.int32)


def parse_decode(spec):
    kind, v = spec.split(':')
    return kind, (float(v) if kind == 'hmax' else int(v))


MIN_SHARE = 1
MIN_AREA = 10
_SE = np.ones((3, 3), bool)


def per_object(gt, pred):
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    out = []
    for v in np.unique(gt):
        if v == 0:
            continue
        gm = gt == v
        ga = int(gm.sum())
        if ga < MIN_AREA:
            continue
        c, n = np.unique(pred[gm], return_counts=True)
        keep = c > 0
        matched, j = None, 0.0
        if keep.any():
            c, n = c[keep], n[keep]
            i = int(n.argmax())
            if n[i] > .5 * ga:
                matched = int(c[i])
                j = float(n[i] / (ga + int((pred == c[i]).sum()) - n[i]))
        ring = ndi.binary_dilation(gm, _SE, 1) & ~gm
        nb = gt[ring]
        nb = nb[nb > 0]
        touching = any(int((ring & (gt == int(u))).sum()) >= MIN_SHARE for u in np.unique(nb))
        out.append(dict(id=int(v), jacc=j, area=ga, touching=bool(touching), matched=matched))
    return out


def _subset(objs, sub):
    if sub == 'touch':
        return [o for o in objs if o['touching']]
    if sub == 'nontouch':
        return [o for o in objs if not o['touching']]
    return objs


def seg_pooled(objs_by_frame, sub='all'):
    j = [o['jacc'] for objs in objs_by_frame.values() for o in _subset(objs, sub)]
    return float(np.mean(j)) if j else float('nan')


def seg_per_image(objs_by_frame, sub='all'):
    per = [np.mean([o['jacc'] for o in _subset(objs, sub)]) for objs in objs_by_frame.values() if _subset(objs, sub)]
    return (float(np.mean(per)), float(np.std(per))) if per else (float('nan'), float('nan'))


def seg_per_image_seeds(runs, sub='all'):
    per = [{t: np.mean([o['jacc'] for o in _subset(objs, sub)]) for t, objs in r.items() if _subset(objs, sub)}
           for r in runs]
    frames = sorted(set.intersection(*[set(p) for p in per]))
    avg = np.array([np.mean([p[t] for p in per]) for t in frames])
    return float(avg.mean()), float(avg.std())


def evaluate(pairs, predict):
    res = collections.OrderedDict()
    for gt, pred, t in pairs:
        res[t] = per_object(gt, pred() if callable(pred) else pred)
    return res
