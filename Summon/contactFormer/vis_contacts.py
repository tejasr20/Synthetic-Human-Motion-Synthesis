import os
import argparse
import torch
import numpy as np
import open3d as o3d
import trimesh
from tqdm import tqdm

import data_utils as du
import vis_utils as vu

device = "cuda"
dtype = torch.float32
use_semantics = True
no_obj_classes = 42
batch_size = 1
gender = 'neutral'
name="sdf"
pose_seq_name="chair2"
save_cf_dir= ""
save_cfs_dir= ""
default_color = [1.0, 0.75, 0.8]

def main():
    parser = argparse.ArgumentParser(description="Process vertices data.")
    parser.add_argument("--vertices-path", type=str, help="Path to the vertices numpy file.")
    # parser.add_argument("--vertices-can-path", type=str, help="Path to the canonical vertices numpy file.")
    parser.add_argument("--save-cf-dir", type=str, help="Directory to save cf data.")
    # parser.add_argument("--save-cfs-dir", type=str, help="Directory to save cf semantics data.")
    parser.add_argument("--sdf-dir", type=str, help="Directory to save cf semantics data.")
    args = parser.parse_args()

    vertices_path = args.vertices_path
    # vertices_can_path = args.vertices_can_path
    save_cf_dir = args.save_cf_dir
    # save_cfs_dir = args.save_cfs_dir
    save_cfs_dir= save_cf_dir
    # Load vertices data
    vertices = np.load(vertices_path)
    # vertices_can = np.load(vertices_can_path)

    scene_data = du.load_scene_data(device=device, name=name, sdf_dir=args.sdf_dir,
                                    use_semantics=use_semantics, no_obj_classes=no_obj_classes)

    cf_list = []
    cf_semantics_list = []

    for i in tqdm(range(vertices.shape[0])):
        vertex = vertices[i]
        # vertex_can = vertices_can[i]

        cf = du.read_sdf(vertex, scene_data['sdf'],
                        scene_data['grid_dim'], scene_data['grid_min'], scene_data['grid_max'],
                        mode="bilinear").squeeze()
        temp = torch.zeros_like(cf)
        temp[cf < 0.05] = 1
        cf = temp
        cf_semantics = du.read_sdf(vertex, scene_data['semantics'],
                                    scene_data['grid_dim'], scene_data['grid_min'], scene_data['grid_max'],
                                    mode="nearest").squeeze()
        cf_semantics[cf != 1] = 0

        cf_list.append(cf)
        cf_semantics_list.append(cf_semantics)

    cf_array = np.array(cf_list)
    cf_semantics_array = np.array(cf_semantics_list)

    np.save(os.path.join(save_cf_dir, f"{pose_seq_name}_cf"), cf_array)
    np.save(os.path.join(save_cfs_dir, f"{pose_seq_name}_cfs"), cf_semantics_array)

if __name__ == "__main__":
    main()
