import torch
import glob
import re
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

model_dir = 'opencood/logs/point_pillar_v2vnet_VN_nll_2026_08_11_19_12_05/'

hypes = yaml_utils.load_yaml(None, type('opt', (), {'model_dir': model_dir})())

print('Building dataset...')
opencood_dataset = build_dataset(hypes, visualize=False, train=False)
data_loader = DataLoader(opencood_dataset,
                          batch_size=1,
                          num_workers=0,
                          collate_fn=opencood_dataset.collate_batch_test,
                          shuffle=False)

print('Creating model...')
model = train_utils.create_model(hypes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Grab one fixed batch to reuse across all checkpoints (keeps comparison apples-to-apples)
batch_data = next(iter(data_loader))
batch_data = train_utils.to_device(batch_data, device)

# Find all checkpoints, sorted by epoch number
ckpt_files = glob.glob(model_dir + 'net_epoch*.pth')
def epoch_num(path):
    return int(re.search(r'net_epoch(\d+)\.pth', path).group(1))
ckpt_files = sorted(ckpt_files, key=epoch_num)

print(f'\n{"Epoch":>6} | {"logit min":>10} | {"logit max":>10} | {"sigmoid max":>12} | {"anchors>0.20":>13}')
print('-' * 65)

for ckpt_path in ckpt_files:
    ep = epoch_num(ckpt_path)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        output_dict = model(batch_data['ego'])

    psm = output_dict['psm']
    probs = torch.sigmoid(psm)
    n_above = (probs > 0.20).sum().item()

    print(f'{ep:>6} | {psm.min().item():>10.4f} | {psm.max().item():>10.4f} | {probs.max().item():>12.6f} | {n_above:>13}')