import torch
import torch.optim as optim
import os
import shutil
import argparse
from tqdm import tqdm
import time, datetime
from collections import defaultdict
import pandas as pd
from src import config
from src.checkpoints import CheckpointIO
from src.utils.io import export_pointcloud
from src.utils.visualize import visualize_data
from src.utils.voxels import VoxelGrid
from tensorboardX import SummaryWriter

import numpy as np
import open3d as o3d



# class Cfg(object):
#       def __init__(self) -> None:
#             self.points_subsample = self.imp_value * 2
#             return
      
# class MoreCfg(Cfg1, Cfg2, ...):
#       def __init__(self) -> None:
#             self.imp_value = 3
#             Cfg.__init__(self)

#             return


class Cfg(object):
      def __init__(self) -> None:
            self.training={
                  'out_dir':"out/demo_syn_room",

            }
            self.generation={
                  'generation_dir':"generation",
                  'vis_n_outputs':30,

            }
            self.data={
                  'input_type':"img",

            }
            self.test={
                  'model_file':"room_grid64.pt"
            }

            return 
      

cfg = Cfg()



parser = argparse.ArgumentParser(
    description='Extract meshes from occupancy process.'
)
#parser.add_argument('config', type=str, help='Path to config file.')

#args = parser.parse_args()
#cfg = config.load_config(args.config, 'configs/default.yaml')

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

out_dir = cfg.training['out_dir']
generation_dir = os.path.join(out_dir, cfg.generation['generation_dir'])

input_type = cfg.data['input_type']
vis_n_outputs = cfg.generation['vis_n_outputs']
if vis_n_outputs is None:
    vis_n_outputs = -1

point_cloud_file_path="/home/js/airplane_pcd.ply"
point_cloud = o3d.io.read_point_cloud(point_cloud_file_path)
point_cloud.estimate_normals()

points = np.asarray(point_cloud.points) # nx3 array
normals = np.asarray(point_cloud.normals)

center = np.mean(points, axis=0)
points -= center
norms = np.linalg.norm(points, axis=1)
max_norm = np.max(norms)
points /= max_norm * 2.0

data = {
    'idx':0,
    'inputs': torch.from_numpy(points).type(torch.float32).cuda().unsqueeze(0),
    'normals':torch.from_numpy(normals).type(torch.float32).cuda().unsqueeze(0),
    #'idx': 0,
}

# Model
model = config.get_model(device=device)

checkpoint_io = CheckpointIO(out_dir, model=model)
checkpoint_io.load(cfg.test['model_file'])

# Generator
generator = config.get_generator(model,device)


def generate_mesh_func(data):
        # Generate
        model.eval()
        out = generator.generate_mesh(data)

        return out[0]

mesh_out_file = os.path.join(out_dir, '%s.off' % (0))

mesh = generate_mesh_func(data)

mesh.export(mesh_out_file)