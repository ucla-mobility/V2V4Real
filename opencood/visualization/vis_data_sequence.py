import os
import argparse
from torch.utils.data import DataLoader

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.visualization import vis_utils
from opencood.data_utils.datasets.early_fusion_vis_dataset import \
    EarlyFusionVisDataset


def vis_parser():
    parser = argparse.ArgumentParser(description="data visualization")
    parser.add_argument('--color_mode', type=str, default="v2vreal",
                        help='lidar color rendering mode, e.g. intensity,'
                             'z-value or constant.')
    parser.add_argument('--isSim', action='store_true')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    current_path = os.path.dirname(os.path.realpath(__file__))
    opt = vis_parser()
    params = load_yaml(os.path.join(current_path,
                                    '../hypes_yaml/visualization.yaml'))

    opencda_dataset = EarlyFusionVisDataset(params, visualize=True,
                                            train=False, isSim=opt.isSim)
    data_loader = DataLoader(opencda_dataset, batch_size=1, num_workers=8,
                             collate_fn=opencda_dataset.collate_batch_train,
                             shuffle=False,
                             pin_memory=False)

    vis_utils.visualize_sequence_dataloader(data_loader,
                                            params['postprocess']['order'],
                                            color_mode=opt.color_mode)
