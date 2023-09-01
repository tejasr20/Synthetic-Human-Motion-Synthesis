# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: Vassilis Choutas, vassilis.choutas@tuebingen.mpg.de

import os
import os.path as osp
import sys
import pickle
import argparse
from visualize import vis_utils
import shutil

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from smplx import build_layer
from typing import List, Tuple

import torch.utils.data as dutils
from .data.datasets import NewMeshFolder

from loguru import logger
from .config import parse_args
from .data import dataloader_new
from .transfer_model import run_fitting
from .utils import read_deformation_transfer, np_mesh_to_o3d
from visualize.smpl_dict import return_smpl_dict
import merged_op_new
from pickle_amass_vertices import pickle_amass_vertices


def dataloader_new(exp_cfg, dict):
    batch_size = exp_cfg.batch_size
    num_workers = exp_cfg.datasets.num_workers
    dataset= NewMeshFolder(dict)
    logger.info(
        f'Creating dataloader with B={batch_size}, workers={num_workers}')
    # batch size is one, so quite irrelevant. 
    dataloader = dutils.DataLoader(dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   shuffle=False)

    return {'dataloader': dataloader, 'dataset': dataset}


def main() -> None:
    exp_cfg = parse_args()
    smpl_dict= return_smpl_dict(exp_cfg.ip_path, exp_cfg.cuda, exp_cfg.device)
    if torch.cuda.is_available() and exp_cfg["use_cuda"]:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        if exp_cfg["use_cuda"]:
            if input("use_cuda=True and GPU is not available, using CPU instead,"
                     " would you like to continue? (y/n)") != "y":
                sys.exit(3)

    logger.remove()
    logger.add(
        lambda x: tqdm.write(x, end=''), level=exp_cfg.logger_level.upper(),
        colorize=True)

    output_folder = osp.expanduser(osp.expandvars(exp_cfg.output_folder))
    logger.info(f'Saving output to: {output_folder}')
    os.makedirs(output_folder, exist_ok=True)

    model_path = exp_cfg.body_model.folder
    body_model = build_layer(model_path, **exp_cfg.body_model)
    logger.info(body_model)
    body_model = body_model.to(device=device)

    deformation_transfer_path = exp_cfg.get('deformation_transfer_path', '')
    def_matrix = read_deformation_transfer(
        deformation_transfer_path, device=device)

    # Read mask for valid vertex ids
    mask_ids_fname = osp.expandvars(exp_cfg.mask_ids_fname)
    mask_ids = None
    if osp.exists(mask_ids_fname):
        logger.info(f'Loading mask ids from: {mask_ids_fname}')
        mask_ids = np.load(mask_ids_fname)
        mask_ids = torch.from_numpy(mask_ids).to(device=device)
    else:
        logger.warning(f'Mask ids fname not found: {mask_ids_fname}')

    # data_obj_dict = build_dataloader(exp_cfg)
    data_obj_dict= dataloader_new(exp_cfg, smpl_dict)
 
    dataloader = data_obj_dict['dataloader']
    # Creates a dataloader over the meshes you have given to convert. 
    data_list=[]
    for ii, batch in enumerate(tqdm(dataloader)):
	#The enumerate() function is a built-in Python function that adds a counter to an iterable object (in this case, the dataloader). It returns an iterator that generates pairs of the form (index, item) for each item in the iterable
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device=device)
                # print("The key in the batch is "+ key)
                # print("Tensor shape is ", batch[key].shape)
        # There are three keys in the batch when converting SMPL to SMPL-X
        # Their shapes are as follows: 'vertices': [1, 6890, 3], 'faces': [1, 13776, 3], and 'indices' : [1]
        print(batch['indices'])
        var_dict = run_fitting(
            exp_cfg, batch, body_model, def_matrix, mask_ids)
        data_list.append(var_dict)
        # paths = batch['paths'] 
    
    data_dict= merged_op_new.merge("", "neutral", data_list)
    pickle_amass_vertices(data_dict, "ARGH", "new", "body_models/", "cuda")
    
        # for ii, path in enumerate(paths):
        #     _, fname = osp.split(path)

            # output_path = osp.join(
            #     output_folder, f'{osp.splitext(fname)[0]}.pkl')
            # with open(output_path, 'wb') as f:
            #     pickle.dump(var_dict, f)

            # output_path = osp.join(
            #     output_folder, f'{osp.splitext(fname)[0]}.obj')
            # mesh = np_mesh_to_o3d(
            #     var_dict['vertices'][ii], var_dict['faces'])
            # print(type(mesh))
            # for key in mesh:
            #     print(key)
            # o3d.io.write_triangle_mesh(output_path, mesh)


if __name__ == '__main__':
    main()
