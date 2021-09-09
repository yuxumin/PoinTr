import os
import yaml
from datasets import build_dataset_from_cfg

from easydict import EasyDict
import argparse
from extensions.chamfer_dist import ChamferDistanceL2_split, ChamferDistanceL2
import torch
import numpy as np

from tqdm import tqdm

def build_ShapeNetCars():
    ShapeNetCars_config = val = yaml.load(open('cfgs/dataset_configs/PCNCars.yaml', 'r'), Loader=yaml.FullLoader)
    train_dataset = build_dataset_from_cfg(EasyDict(ShapeNetCars_config), EasyDict(subset = 'train'))
    test_dataset = build_dataset_from_cfg(EasyDict(ShapeNetCars_config), EasyDict(subset = 'test'))
    val_dataset = build_dataset_from_cfg(EasyDict(ShapeNetCars_config), EasyDict(subset = 'val'))
    CarsDataset = train_dataset + test_dataset + val_dataset
    return CarsDataset
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vis_path', 
        type = str, 
        help = 'KITTI visualize path')
    args = parser.parse_args()
    return args

def get_Fidelity():
    # Fidelity Error
    criterion = ChamferDistanceL2_split(ignore_zeros=True)

    metric = []
    for sample in Samples:
        input_data = torch.from_numpy(np.load(os.path.join(Data_path, sample, 'input.npy'))).unsqueeze(0).cuda()
        pred_data = torch.from_numpy(np.load(os.path.join(Data_path, sample, 'pred.npy'))).unsqueeze(0).cuda()
        metric.append(criterion(input_data, pred_data)[0])
    print('Fidelity is %f' % (sum(metric)/len(metric)))

def get_Consistency():
    #Consistency
    criterion = ChamferDistanceL2(ignore_zeros=True)
    Cars_dict = {}
    for sample in Samples:
        all_elements = sample.split('_') # example sample = 'frame_1_car_3_647'
        frame_id = int(all_elements[1])
        car_id = int(all_elements[-2])
        sample_id = int(all_elements[-1])

        if Cars_dict.get(car_id) is None:
            Cars_dict[car_id] = [f'frame_{frame_id:03d}_car_{car_id:02d}_{sample_id:03d}']
        else:
            Cars_dict[car_id].append(f'frame_{frame_id:03d}_car_{car_id:02d}_{sample_id:03d}') # example sample = 'frame_001_car_003_647'
    
    Consistency = []
    for key, car_list in Cars_dict.items():
        car_list = sorted(car_list)
        Each_Car_Consistency = []
        for i, this_car in enumerate(car_list):
            if i == len(car_list) - 1:
                break
            this_elements = this_car.split('_')
            this_frame =int(this_elements[1])
            
            next_car = car_list[i + 1]
            next_elements = next_car.split('_')
            next_frame = int(next_elements[1])
            
            if next_frame - 1 != this_frame:
                continue
            
            this_car = torch.from_numpy(np.load(os.path.join(Data_path, f'frame_{this_frame}_car_{int(this_elements[3])}_{int(this_elements[4]):03d}', 'pred.npy'))).unsqueeze(0).cuda()
            next_car = torch.from_numpy(np.load(os.path.join(Data_path, f'frame_{next_frame}_car_{int(next_elements[3])}_{int(next_elements[4]):03d}', 'pred.npy'))).unsqueeze(0).cuda()
            cd = criterion(this_car, next_car)
            Each_Car_Consistency.append(cd)
        
        MeanCD = sum(Each_Car_Consistency) / len(Each_Car_Consistency)
        Consistency.append(MeanCD)
    MeanCD = sum(Consistency) / len(Consistency)
    print(f'Consistency is {MeanCD:.6f}')

def get_MMD():
    criterion = ChamferDistanceL2(ignore_zeros=True)
    #MMD
    metric = []
    for item in tqdm(sorted(Samples)):
        pred_data = torch.from_numpy(np.load(os.path.join(Data_path, item, 'pred.npy'))).unsqueeze(0).cuda()
        batch_cd = []
        for index in range(len(ShapeNetCars_dataset)):
            gt = ShapeNetCars_dataset[index][-1][1].cuda().unsqueeze(0)
        # for index, (taxonomy_ids, model_ids, data) in enumerate(CarsDataloader):
            # gt = data[1].cuda()
            # batch_pred_data = pred_data.expand(gt.size(0), -1, -1).contiguous()
            min_cd = criterion(gt, pred_data)
            batch_cd.append(min_cd)
        min_cd = min(batch_cd).item()
        metric.append(min_cd)
        print('This item %s CD %f, MMD %f' % (item, min_cd, sum(metric)*1.0 / len(metric) ))
    print('MMD is %f' % (sum(metric)/len(metric)))

if __name__ == '__main__':
    args = get_args()
    ShapeNetCars_dataset = build_ShapeNetCars()
    CarsDataloader = torch.utils.data.DataLoader(ShapeNetCars_dataset, batch_size=1, shuffle = False, num_workers = 8)

    # Your data
    Data_path = args.vis_path
    Samples = [item for item in os.listdir(Data_path) if os.path.isdir(Data_path + '/' + item)]
    criterion = ChamferDistanceL2_split(ignore_zeros=True)

    get_Fidelity()
    get_MMD()
