import torch.utils.data as data
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import json
from .build import DATASETS

# References:
# - https://github.com/hzxie/GRNet/blob/master/utils/data_loaders.py
@DATASETS.register_module()
class KITTI(data.Dataset):
    def __init__(self, config):
        self.cloud_path = config.CLOUD_PATH
        self.bbox_path = config.BBOX_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        assert self.subset == 'test'

        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
        self.transforms = data_transforms.Compose([{
                'callback': 'NormalizeObjectPose',
                'parameters': {
                    'input_keys': {
                        'ptcloud': 'partial_cloud',
                        'bbox': 'bounding_box'
                    }
                },
                'objects': ['partial_cloud', 'bounding_box']
            }, {
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'bounding_box']
            }])
        self.file_list = self._get_file_list(self.subset)

    def _get_file_list(self, subset):
        """Prepare file list for the dataset"""
        file_list = []
        for dc in self.dataset_categories:
            samples = dc[subset]
            for s in samples:
                file_list.append({
                    'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_cloud_path': self.cloud_path % s,
                    'bounding_box_path': self.bbox_path % s,
                })
        return file_list

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}

        for ri in ['partial_cloud', 'bounding_box']: 
            file_path = sample['%s_path' % ri]
            data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        return  sample['taxonomy_id'], sample['model_id'], data['partial_cloud']
