"""Detection-only (reg + conf) OPV2V validation loss per checkpoint.

Mirrors the validation loop in opencood/tools/train.py. For VN models the
uncertainty term is left out, so the value is comparable to the baseline loss.
Usage: python opencood/tools/det_val_loss.py <run_dir> <first_epoch> <last_epoch> <out.csv>
"""
import os
import statistics
import sys

import torch
from torch.utils.data import DataLoader

from opencood.data_utils.datasets import build_dataset
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils

run_dir, e0, e1, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
hypes = load_yaml(os.path.join(run_dir, 'config.yaml'))
# Always validate on OPV2V (baseline config.yaml was edited to V2V4Real for testing).
hypes['dataset'] = 'opv2v'
hypes['validate_dir'] = '/data/s3/aziya_master_workspace/opv2v_data/validate'

ds = build_dataset(hypes, visualize=False, train=False)
loader = DataLoader(ds, batch_size=hypes['train_params']['batch_size'],
                    num_workers=8, collate_fn=ds.collate_batch_train,
                    shuffle=False, pin_memory=False, drop_last=True)
device = torch.device('cuda')
model = train_utils.create_model(hypes).to(device)
criterion = train_utils.create_loss(hypes)

done = set()
if os.path.exists(out):
    done = {int(l.split(',')[0]) for l in open(out).read().split('\n')[1:] if l}
else:
    open(out, 'w').write('ckpt_epoch,det_loss,reg_loss,conf_loss,total_loss\n')

for ep in range(e0, e1 + 1):
    path = os.path.join(run_dir, 'net_epoch%d.pth' % ep)
    if ep in done or not os.path.exists(path):
        continue
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    reg, conf, tot = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = train_utils.to_device(batch, device)
            o = model(batch['ego'])
            t = criterion(o, batch['ego']['label_dict'])
            reg.append(criterion.loss_dict['reg_loss'].item())
            conf.append(criterion.loss_dict['conf_loss'].item())
            tot.append(t.item())
    r, c, t = map(statistics.mean, (reg, conf, tot))
    line = '%d,%.6f,%.6f,%.6f,%.6f' % (ep, r + c, r, c, t)
    open(out, 'a').write(line + '\n')
    print(line, flush=True)
