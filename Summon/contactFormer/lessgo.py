import os
import argparse
import torch
import numpy as np
import data_utils as du

device = "cuda"
dtype = torch.float32
batch_size = 1
gender = 'neutral'

smplx_model_path = "../data/body_models/smplx/SMPLX_NEUTRAL.npz"
cam_dir= "../../POSA/cam2world"
posa_cam2world = "../../POSA/cam2world"
posa_scene = "../../POSA/scenes"
test_scene_name = "MPH16"
test_scene_mesh_path = "{}/{}.ply".format(posa_scene, test_scene_name)
test_sdf_dir = "../../POSA/sdf"
test_cam_path = "{}/{}.json".format(posa_cam2world, test_scene_name)
test_pkl_file_path = "../data/PROXD_temp/MPH16_00157_01/results/s001_frame_00228__00.00.07.577/000.pkl"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset for temporal_POSA")
    parser.add_argument("--input_dir", type=str, help="path to directory containing .pkl files")
    parser.add_argument("--output_dir", type=str, help="path to save generated dataset")
    parser.add_argument("--name", type=str, help="name to use for saving the results")

    # Parse arguments and assign directories
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    name = args.name

    save_verts_dir = os.path.join(output_dir, "vertices")
    save_verts_can_dir = os.path.join(output_dir, "vertices_can")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not os.path.isdir(save_verts_dir):
        os.mkdir(save_verts_dir)

    if not os.path.isdir(save_verts_can_dir):
        os.mkdir(save_verts_can_dir)

    pose_seq_path = input_dir
    pkl_files_list = os.listdir(pose_seq_path)
    # pkl_files_list.sort()
    pkl_files_list.sort(key=lambda x: int(x.split('_')[1][:-4]))
    cam_path = os.path.join(cam_dir,"MPH8.json")
    vertices_list = []
    vertices_can_list = []

    for i, f in enumerate(pkl_files_list):
        # print(f)
        # pkl_file_path = os.path.join(pose_seq_path, f, "000.pkl")
        pkl_file_path = os.path.join(pose_seq_path, f)
        # print(pkl_file_path)
        vertices_can, vertices = du.pkl_to_canonical(pkl_file_path, device, dtype, batch_size, gender,
                                                     smplx_model_path, cam_path)

        vertices_list.append(vertices.squeeze().cpu().detach().numpy())
        vertices_can_list.append(vertices_can.squeeze().cpu().detach().numpy())

    vertices_list = np.array(vertices_list)
    vertices_can_list = np.array(vertices_can_list)

    np.save(os.path.join(save_verts_dir, f"{name}_verts.npy"), vertices_list)
    np.save(os.path.join(save_verts_can_dir, f"{name}_verts_can.npy"), vertices_can_list)
    print(f"Saved vertices to {output_dir}")