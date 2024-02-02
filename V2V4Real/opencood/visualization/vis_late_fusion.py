import numpy as np
import open3d as o3d
from torch.utils.data import DataLoader

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.visualization import vis_utils
from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset

if __name__ == '__main__':
    params = load_yaml('../hypes_yaml/voxelnet_late_fusion.yaml')
    opencda_dataset = LateFusionDataset(params, visualize=True, train=False)

    data_loader = DataLoader(opencda_dataset, batch_size=1, num_workers=8,
                             collate_fn=opencda_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False)
    # vis_utils.visualize_sequence_dataloader(data_loader,
    #                                         params['postprocess']['order'])
    o3d_pcd = o3d.geometry.PointCloud()
    for j, batch_data in enumerate(data_loader):
        vis_utils.visualize_single_sample_dataloader(batch_data['ego'],
                                                     o3d_pcd,
                                                     params['postprocess'][
                                                         'order'],
                                                     visualize=True,
                                                     save_path='tmp.png',
                                                     oabb=True)
