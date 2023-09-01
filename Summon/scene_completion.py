import os
import argparse
from pathlib import Path
import json
import numpy as np
import open3d as o3d
import random
import shutil
import torch
from atiss.scripts.training_utils import load_config
from atiss.scene_synthesis.networks import build_network
from utils import write_verts_faces_obj, align_obj_to_floor, fun

object_types = [
    'armchair',
    'bookshelf',
    'cabinet',
    'ceiling_lamp',
    'chair',
    'children_cabinet',
    'coffee_table',
    'desk',
    'double_bed',
    'dressing_chair',
    'dressing_table',
    'kids_bed',
    'nightstand',
    'pendant_lamp',
    'shelf',
    'single_bed',
    'sofa',
    'stool',
    'table',
    'tv_stand',
    'wardrobe',
    'other',
    'none'
]


def get_grid_index(grid_center, grid_half_length, grid_size, point):
    #Given a grid center, grid half length, grid size, and a point, it calculates the index of the cell in the grid that the point belongs to.
    top_left = np.array((grid_center[0] - grid_half_length, grid_center[1] - grid_half_length))
    cell_length = grid_half_length * 2 / grid_size
    offset = point - top_left
    index = np.floor(offset / cell_length).astype(int)
    return np.transpose(index)


def get_cell_center(grid_length, grid_size, cell_index):
    #Given the grid length, grid size, and cell index, it calculates the center coordinates of the cell.
    cell_size = grid_length / grid_size
    return np.array(((cell_index[1] + 0.5) * cell_size, (cell_index[0] + 0.5) * cell_size))


def check_area_occupied(occ_grid, top_left_index, bot_right_index):
    #Given an occupancy grid, top-left index, and bottom-right index, it checks whether the area specified by the indices is occupied in the grid.
    return occ_grid[top_left_index[0]:bot_right_index[0]+1, top_left_index[1]:bot_right_index[1]+1].sum() != 0


def get_obj_list(fitting_results_dir):
    #It retrieves a list of object axis-aligned bounding boxes (AABB) from the fitting results directory.
    obj_list = []
    for obj_class_dir in fitting_results_dir.iterdir():
        for obj_dir in obj_class_dir.iterdir():
            with open(str(obj_dir / 'best_obj_id.json'), "r") as f:
                best_obj_json = json.load(f)
            best_obj_id = best_obj_json['best_obj_id']
            best_obj_path = obj_dir / best_obj_id / 'opt_best.obj'
            obj_mesh = o3d.io.read_triangle_mesh(str(best_obj_path))
            obj_aabbox = obj_mesh.get_axis_aligned_bounding_box()
            obj_list.append(obj_aabbox)
    return obj_list


def get_human_list():
    # It retrieves a list of human axis-aligned bounding boxes(AABBs) from the fitting results directory.
    human_list = []
    human_mesh_dir = Path(args.fitting_results_path) / 'human' / 'mesh'
    for human_mesh_path in sorted(human_mesh_dir.iterdir()):
        human_mesh = o3d.io.read_triangle_mesh(str(human_mesh_path))
        human_aabbox = human_mesh.get_axis_aligned_bounding_box()
        human_list.append(human_aabbox)
    human_list = human_list[::8]
    return human_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--fitting_results_path", type=str, help="Fitting result directory for some sequence")
    parser.add_argument("--obj_dataset_path", type=str, help="Path to the 3D-Future dataset")
    parser.add_argument("--path_to_model", type=str, help="Path to ATISS model checkpoint")
    parser.add_argument("--num_iter", type=int, default=3)
    parser.add_argument("--spare_length", type=float, default=1)
    parser.add_argument("--vis_bb", type=int, default=0)
    # parser.add_argument("--empty", type=int, default= 1, help= "Whether we want to generate without contact objects")
    args = parser.parse_args()
    print(args.fitting_results_path+ "/fit_best_obj")
    # Before adding non-contact objects to the scene, first remove non-contact objects added from last time
    if not os.path.exists(args.fitting_results_path+"/fit_best_obj"):
        os.makedirs(args.fitting_results_path+ "/fit_best_obj")
		# To avoid errors when running directly on human meshes. 
    fitting_results_dir = Path(args.fitting_results_path) / 'fit_best_obj'
    for obj_class_dir in fitting_results_dir.iterdir():
        # if(args.empty==0):
        #     break
        for obj_dir in obj_class_dir.iterdir():
            with open(str(obj_dir / 'best_obj_id.json'), "r") as f:
                best_obj_json = json.load(f)
            if 'no_contact' in best_obj_json:
                shutil.rmtree(str(obj_dir))
        if len(os.listdir(str(obj_class_dir))) == 0:
            obj_class_dir.rmdir()

    '''
    Start adding objects
    '''
    obj_dataset_path = Path(args.obj_dataset_path)
    # Setup ATISS model
    device = torch.device("cpu")
    config_path = os.path.join("atiss", "config", "bedrooms_eval_config.yaml")
    config = load_config(config_path)
    
    
    weight_file = args.path_to_model
    network, _, _ = build_network(
        30, 23,
        config, weight_file, device=device
    )
    network.eval()
	# get_center() Returns the center of the geometry coordinates: numpy.ndarray[numpy.float64[3, 1]]
    '''
    Get the scene center and scene length based on contact objects and human motion
    '''
    obj_list = get_obj_list(fitting_results_dir)
    human_list = get_human_list()
    # Get the virtual scene center
    total_obj_list = obj_list + human_list
    scene_center = np.zeros((3,))
    for bbox in total_obj_list:
        scene_center += bbox.get_center()
    scene_center /= len(total_obj_list)
    # Get the size of the virtual scene (assuming virtual scene is square)
    scene_length = 0
    # Loop calculates the maximum distance between the center of a bounding box and the scene center. 
    # It then compares this distance to the current maximum scene length and updates the scene length 
    # if the calculated distance is larger.
    for bbox in total_obj_list:
        half_extent = bbox.get_half_extent() # Returns the half extent of the bounding box: numpy.ndarray[numpy.float64[3, 1]]
        offset = np.abs(bbox.get_center() - scene_center) # distance of each bounding boxes centre from the scene centre 
        cur_max_dist = np.max(offset[:2]) + np.max(half_extent[:2]) 
        #np.max(offset[:2]) calculates the maximum distance in the x and y dimensions by taking the maximum values of the first two elements of the offset array.
        cur_max_dist *= 2
        if cur_max_dist > scene_length:
            scene_length = cur_max_dist
    scene_length += args.spare_length
    print("Scene length is ", scene_length)        
    print("Scene center is ", scene_center)
    # add an arbitrary extra value to the scene size
    
    # scene_length/=2
    # print(scene_length)
    # scene_center= 2
    if(args.vis_bb):
        fun(scene_center= scene_center, total_obj_list= total_obj_list)
    for iter in range(args.num_iter): # to add num_iters items. 
        print(f"Currently trying to add #{iter + 1} item")

        # Get the bounding box for all existed objects
        obj_list = get_obj_list(fitting_results_dir) # This includes newly added objects. 
        # Update total_obj_list
        total_obj_list = obj_list + human_list
        
        # recalculation of scene length and center to improve calculations : added by me 
        # scene_center= np.zeros((3,))
        # for bbox in total_obj_list:
        #     scene_center += bbox.get_center()
            
        # scene_center /= len(total_obj_list)
        # scene_length=0
        # for bbox in total_obj_list:
        #     half_extent = bbox.get_half_extent() # Returns the half extent of the bounding box: numpy.ndarray[numpy.float64[3, 1]]
        #     offset = np.abs(bbox.get_center() - scene_center) # distance of each bounding boxes centre from the scene centre 
        #     cur_max_dist = np.max(offset[:2]) + np.max(half_extent[:2]) 
        #     #np.max(offset[:2]) calculates the maximum distance in the x and y dimensions by taking the maximum values of the first two elements of the offset array.
        #     cur_max_dist *= 2
        #     if cur_max_dist > scene_length:
        #         scene_length = cur_max_dist
        # scene_length += args.spare_length
        # scene_length/=2
        '''
        Get the class distribution for next potential object using ATISS
        '''
        # Get input (i.e. boxes) for ATISS model
        num_obj = len(obj_list)
        boxes = {}
        boxes['class_labels'] = torch.zeros((1, num_obj, 23)).to(device) # 23 classes of objects to be added. One hot. 
        boxes['translations'] = torch.zeros((1, num_obj, 3)).to(device)
        boxes['sizes'] = torch.zeros((1, num_obj, 3)).to(device)
        boxes['angles'] = torch.zeros((1, num_obj, 1)).to(device)
        # Fill in input boxes attributes
        item_idx = 0
        for obj_class_dir in fitting_results_dir.iterdir():
            for obj_dir in obj_class_dir.iterdir():
                obj_class = obj_class_dir.stem # Gets name of object 
                obj_class_idx = object_types.index(obj_class)
                boxes['class_labels'][0, item_idx, obj_class_idx] = 1
                item_idx += 1
                # TODO: to get a better estimation of next class distribution, we shall fill in translations/angles/sizes
        # Get next object class distribution
        room_mask = torch.ones((1, 1, 64, 64)).to(device)
        class_prob = network.distribution_classes(boxes, room_mask)
        class_prob = class_prob.squeeze().detach().numpy()

        # Build an occupation grid
        # grid_size = 256
        grid_size = 100# trying new values 
        occ_grid = np.zeros((grid_size, grid_size))
        # Fill the occupation grid with existing objects and human meshes
        # iterates over each bounding box in total_obj_list and calculates the top-left
        # and bottom-right coordinates of the bounding box in 2D space.
        for bbox in total_obj_list:
            center = bbox.get_center()
            half_extent = bbox.get_half_extent()
            top_left = (center - half_extent)[:2]
            bot_right = (center + half_extent)[:2]
            top_left_index = get_grid_index(grid_center=scene_center, grid_half_length=scene_length / 2, grid_size=grid_size, point=top_left)
            bot_right_index = get_grid_index(grid_center=scene_center, grid_half_length=scene_length / 2, grid_size=grid_size, point=bot_right)
            occ_grid[top_left_index[0]:bot_right_index[0]+1, top_left_index[1]:bot_right_index[1]+1] = 1
        
        '''
        Sample new object category
        '''
        sampled_class = None
        while(True):
            sampled_class = np.random.choice(len(object_types), p=class_prob)
            sampled_class = object_types[sampled_class]
            if (obj_dataset_path / sampled_class).exists():
                break

        # Get the list of all available objects
        print(f"Sampled next object class is {sampled_class}")
        sampled_class_dir = obj_dataset_path / sampled_class
        new_obj_list = []
        for new_obj_path in sampled_class_dir.iterdir():
            new_obj_list.append(new_obj_path)
        if len(new_obj_list) > 3:
            new_obj_list = np.random.choice(new_obj_list, size=3)

        # Randomly choose an object from candidate list and try to place the object in unoccupied area
        added_obj = False
        for new_obj_path in new_obj_list:
            new_obj_id = new_obj_path.stem
            new_obj_path = new_obj_path / 'raw_model.obj'
            new_obj_mesh = o3d.io.read_triangle_mesh(str(new_obj_path))
            new_obj_bbox = new_obj_mesh.get_axis_aligned_bounding_box()
            new_obj_half_extent = new_obj_bbox.get_half_extent()
            candidate_indices = []
            for i in range(grid_size):
                for j in range(grid_size):
                    if occ_grid[i, j] == 0:
                        cell_center = get_cell_center(scene_length, grid_size, (i, j))
                        new_top_left = cell_center - new_obj_half_extent[:2]
                        new_bot_right = cell_center + new_obj_half_extent[:2]
                        new_top_left_index = get_grid_index(grid_center=scene_center, grid_half_length=scene_length / 2,
                                                            grid_size=grid_size, point=new_top_left)
                        new_bot_right_index = get_grid_index(grid_center=scene_center, grid_half_length=scene_length / 2,
                                                            grid_size=grid_size, point=new_bot_right)
                        if not check_area_occupied(occ_grid, new_top_left_index, new_bot_right_index):
                            candidate_indices.append((i, j))

            # Choose a candidate position randomly and place the object
            if len(candidate_indices) != 0:
                chosen_index = random.choice(candidate_indices)
                # Get faces
                new_obj_faces = np.array(new_obj_mesh.triangles)
                # Get floor-aligned object
                old_obj_verts = np.array(new_obj_mesh.vertices)
                old_obj_verts = align_obj_to_floor(old_obj_verts, new_obj_faces)
                old_obj_center = np.mean(old_obj_verts, axis=0)
                new_obj_center = get_cell_center(scene_length, grid_size, chosen_index)
                new_obj_center = np.array((new_obj_center[0], new_obj_center[1], old_obj_center[2]))
                new_obj_verts = old_obj_verts - old_obj_center + new_obj_center
                # Save object
                obj_save_dir = fitting_results_dir / sampled_class
                if obj_save_dir.exists():
                    num_existed_obj = 0
                    for _ in obj_save_dir.iterdir():
                        num_existed_obj += 1
                    obj_save_path = obj_save_dir / str(num_existed_obj)
                else:
                    obj_save_dir.mkdir()
                    obj_save_path = obj_save_dir / '0'
                obj_save_path.mkdir()
                obj_save_mesh_path = obj_save_path / new_obj_id
                obj_save_mesh_path.mkdir()
                obj_save_mesh_path = obj_save_mesh_path / 'opt_best.obj'
                write_verts_faces_obj(new_obj_verts, new_obj_faces, obj_save_mesh_path)
                obj_save_json_path = obj_save_path / 'best_obj_id.json'
                json_dict = {'best_obj_id': new_obj_id, 'no_contact': True}
                json.dump(json_dict, open(str(obj_save_json_path), 'w'))
                added_obj = True
                break

        if not added_obj:
            print(f"Failed to add any new object of class {sampled_class} due to the scene limitation.")
            
            
            
# best for circle_dir : --spare_length -4 --num_iter 2