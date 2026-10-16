import argparse
import glob
import os
import time
import types

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from dataset import AugSet, gapfill, get_pairs, jitter, md_rule, norm_cp, pad_to, worker_init
from models import TILE, FrozenEncoder, TouchSplitHead
from utils import decode, parse_decode, per_object, predict_fields, seg_per_image, seg_per_image_seeds, seg_pooled, tiles


def train_encoder(cfg, a):
    tr = get_pairs(a.ctc_root, a.dataset, a.data_csv, 'train')
    imgs = [tifffile.imread(ip) for ip, _, _, _ in tr]
    labs = [gapfill(tifffile.imread(sp)) for _, sp, _, _ in tr]
    print('%s: %d training frames (%d GT, %d ST), seed %d'
          % (a.dataset, len(tr), sum(k == 'GT' for *_, k in tr), sum(k == 'ST' for *_, k in tr), a.seed))

    from cellpose import models, train as cptrain
    rng = np.random.default_rng(a.seed)
    orig = cptrain.random_rotate_and_resize

    def augment(*args, **kw):
        out = orig(*args, **kw)
        for n in range(out[0].shape[0]):
            out[0][n] = jitter(out[0][n], rng)
        return out

    cptrain.random_rotate_and_resize = augment

    class _Rand(types.ModuleType):
        def __getattr__(self, k):
            return getattr(np.random, k)

        def seed(self, x=None):
            np.random.seed(None if x is None else a.seed * 10000 + int(x))

    class _NP(types.ModuleType):
        def __getattr__(self, k):
            return getattr(np, k)

    shim = _NP('np_seeded')
    shim.random = _Rand('np_seeded.random')
    cptrain.np = shim

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    name = 'cellpose_ft_%s_s%d' % (a.dataset, a.seed)
    model = models.CellposeModel(gpu=True)
    cptrain.train_seg(model.net, train_data=imgs, train_labels=labs, n_epochs=cfg['encoder']['epochs'],
                      model_name=name, save_path=a.out)
    path = os.path.join(a.out, name + '.pt')
    torch.save(model.net.state_dict(), path)
    print('saved', path)
    return path


def train_decoder(cfg, a, encoder):
    c = cfg['decoder']
    tr = get_pairs(a.ctc_root, a.dataset, a.data_csv, 'train')
    images = [norm_cp(pad_to(tifffile.imread(ip), TILE, TILE)) for ip, _, _, _ in tr]
    labels = [pad_to(gapfill(tifffile.imread(sp)), TILE, TILE) for _, sp, _, _ in tr]
    n = sum(len(tiles(*im.shape, st=128)) for im in images) // c['batch_size'] * c['batch_size']
    print('%s: %d frames, %d samples / epoch, lambda %g, lr %g, seed %d'
          % (a.dataset, len(tr), n, c['lam'], c['lr'], a.seed))

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    enc = FrozenEncoder(encoder)
    head = TouchSplitHead().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c['epochs'])
    bce = nn.BCEWithLogitsLoss()
    dl = torch.utils.data.DataLoader(AugSet(images, labels, n, a.seed), batch_size=c['batch_size'], shuffle=False,
                                     num_workers=c['workers'], drop_last=True, worker_init_fn=worker_init)
    t0 = time.time()
    for ep in range(c['epochs']):
        head.train()
        tot = nb = 0
        for x, z in dl:
            z = z.cuda()
            sides, fu, d = head(enc.tokens(x))
            loss = sum(bce(o, z[:, 0]) for o in sides) / len(sides) + bce(fu, z[:, 0]) + c['lam'] * F.mse_loss(d, z[:, 1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sch.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            print('  epoch %d  loss %.4f  [%.1f min]' % (ep + 1, tot / nb, (time.time() - t0) / 60), flush=True)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, 'touchsplit_head_%s_s%d.pt' % (a.dataset, a.seed))
    torch.save(head.state_dict(), path)
    print('saved', path)


def load_head(path):
    head = TouchSplitHead().cuda()
    head.load_state_dict(torch.load(path, map_location='cuda'))
    return head


def cmd_train(cfg, a):
    encoder = a.encoder or train_encoder(cfg, a)
    if not a.encoder_only:
        train_decoder(cfg, a, encoder)


def cmd_select(cfg, a):
    tr = get_pairs(a.ctc_root, a.dataset, a.data_csv, 'train')
    md = md_rule([pad_to(gapfill(tifffile.imread(sp)), TILE, TILE) for _, sp, _, _ in tr])
    grid = [('md', m) for m in sorted({max(2, int(round(md * r))) for r in (1.0, 0.75, 0.5)}, reverse=True)]
    grid += [('hmax', h) for h in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)]
    gts = {t: tifffile.imread(sp).astype(np.int32) for _, sp, t, _ in tr}
    enc = FrozenEncoder(a.encoder)
    scores = {g: [] for g in grid}
    for hp in a.weights:
        head = load_head(hp)
        fields = {t: predict_fields(tifffile.imread(ip), enc, head) for ip, _, t, _ in tr}
        for g in grid:
            res = {t: per_object(gts[t], decode(fields[t], *g)) for t in fields}
            scores[g].append(seg_per_image(res)[0])
    ranked = sorted(grid, key=lambda g: -np.mean(scores[g]))
    for g in ranked:
        print('  %-4s %-5g  train per-image SEG %.4f' % (g[0], g[1], np.mean(scores[g])))
    print('selected: %s:%g' % ranked[0])


def cmd_predict(cfg, a):
    spec = a.decode or cfg['decode'].get(a.dataset)
    if spec is None:
        raise SystemExit('no marker parameter for %s: pass --decode or add it to the config' % a.dataset)
    enc = FrozenEncoder(a.encoder)
    head = load_head(a.weights[0])
    kind, value = parse_decode(spec)
    os.makedirs(a.output, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.input, '*.tif')))
    for f in files:
        lab = decode(predict_fields(tifffile.imread(f), enc, head), kind, value)
        tifffile.imwrite(os.path.join(a.output, 'mask' + os.path.basename(f).lstrip('t')), lab.astype(np.uint16))
    print('wrote %d label images to %s (%s)' % (len(files), a.output, spec))


def load_run(pairs, folder):
    res = {}
    for ip, sp, t, _ in pairs:
        name = os.path.basename(ip)
        pred = tifffile.imread(os.path.join(folder, 'mask' + (name[1:] if name.startswith('t') else name)))
        res[t] = per_object(tifffile.imread(sp), pred)
    return res


def cmd_evaluate(cfg, a):
    test = cfg['data']['test']
    pairs = get_pairs(a.ctc_root, a.dataset, a.data_csv, 'test', test)
    runs = [load_run(pairs, p) for p in a.pred]
    print('%s %s: %d frames, %d seed(s)' % (a.dataset, test, len(pairs), len(runs)))
    for sub, name in (('all', 'SEG'), ('touch', 'Touch SEG'), ('nontouch', 'Nontouch SEG')):
        m, s = seg_per_image_seeds(runs, sub)
        pooled = sum(seg_pooled(r, sub) for r in runs) / len(runs)
        print('  %-13s per-image %.3f (%.3f)   object-pooled %.4f' % (name, m, s, pooled))


def main():
    ap = argparse.ArgumentParser(description='TouchSplit: train, select markers, predict and evaluate.')
    ap.add_argument('mode', choices=['train', 'select', 'predict', 'evaluate'])
    ap.add_argument('-c', '--config', default='touchsplit.yml')
    ap.add_argument('-d', '--dataset', help='CTC dataset name, or a label for CSV data (default: config)')
    ap.add_argument('--ctc-root', help='CTC data folder (default: config)')
    ap.add_argument('--data-csv', help='CSV with image_path, mask_path, set (replaces --ctc-root)')
    ap.add_argument('-s', '--seed', type=int, default=0)
    ap.add_argument('-e', '--encoder', help='fine-tuned CellposeSAM checkpoint; train: skip encoder fine-tuning')
    ap.add_argument('-w', '--weights', nargs='+', help='TouchSplit decoder checkpoint(s); select: one per seed')
    ap.add_argument('--encoder-only', action='store_true', help='train: fine-tune the encoder only')
    ap.add_argument('-i', '--input', help='predict: folder of .tif images')
    ap.add_argument('-o', '--output', help='predict: output folder for label images')
    ap.add_argument('--decode', help="predict: 'hmax:<h>' or 'md:<pixels>' (default: config)")
    ap.add_argument('--pred', nargs='+', help='evaluate: one prediction folder per seed')
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    a.dataset = a.dataset or cfg['data']['dataset']
    a.ctc_root = a.ctc_root or cfg['data'].get('ctc_root')
    a.data_csv = a.data_csv or cfg['data'].get('data_csv')
    a.out = cfg.get('out', 'weights')
    need = {'select': ['encoder', 'weights'], 'predict': ['encoder', 'weights', 'input', 'output'], 'evaluate': ['pred']}
    missing = [k for k in need.get(a.mode, []) if not getattr(a, k)]
    if missing:
        ap.error('%s needs %s' % (a.mode, ', '.join('--' + k for k in missing)))
    {'train': cmd_train, 'select': cmd_select, 'predict': cmd_predict, 'evaluate': cmd_evaluate}[a.mode](cfg, a)


if __name__ == '__main__':
    main()
