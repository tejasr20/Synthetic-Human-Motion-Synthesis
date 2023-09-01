import json
import os
import time

import torch
import torch.nn as nn

import trimesh

import numpy as np
import open3d as o3d

import scipy.sparse


class ds_us(nn.Module):
    """docstring for ds_us."""

    def __init__(self, M):
        super(ds_us, self).__init__()
        self.M = M

    def forward(self, x):
        """Upsample/downsample mesh. X: B*C*N"""
        out = []
        x = x.transpose(1, 2)
        for i in range(x.shape[0]):
            y = x[i]
            y = spmm(self.M, y)
            out.append(y)
        x = torch.stack(out, dim=0)
        return x.transpose(2, 1)


def adjmat_sparse(adjmat, nsize=1):
    """Create row-normalized sparse graph adjacency matrix."""
    adjmat = scipy.sparse.csr_matrix(adjmat)
    if nsize > 1:
        orig_adjmat = adjmat.copy()
        for _ in range(1, nsize):
            adjmat = adjmat * orig_adjmat
    adjmat.data = np.ones_like(adjmat.data)
    for i in range(adjmat.shape[0]):
        adjmat[i, i] = 1
    num_neighbors = np.array(1 / adjmat.sum(axis=-1))
    adjmat = adjmat.multiply(num_neighbors)
    adjmat = scipy.sparse.coo_matrix(adjmat)
    row = adjmat.row
    col = adjmat.col
    data = adjmat.data
    i = torch.LongTensor(np.array([row, col]))
    v = torch.from_numpy(data).float()
    adjmat = torch.sparse.FloatTensor(i, v, adjmat.shape)
    return adjmat


def scipy_to_pytorch(x):
    """Convert scipy sparse matrices to pytorch sparse matrix."""
    x = scipy.sparse.coo_matrix(x)
    i = torch.LongTensor(np.array([x.row, x.col]))
    v = torch.FloatTensor(x.data)
    return torch.sparse.FloatTensor(i, v, x.shape)


def get_graph_params(ds_us_dir, layer, device, **kwargs):
    """Load and process graph adjacency matrix and upsampling/downsampling matrices."""
    A = scipy.sparse.load_npz(os.path.join(ds_us_dir, 'A_{}.npz'.format(layer)))
    D = scipy.sparse.load_npz(os.path.join(ds_us_dir, 'D_{}.npz'.format(layer)))
    U = scipy.sparse.load_npz(os.path.join(ds_us_dir, 'U_{}.npz'.format(layer)))

    D = scipy_to_pytorch(D).to(device)
    U = scipy_to_pytorch(U).to(device)
    A = adjmat_sparse(A).to(device)
    return A, U, D

    
class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """

    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input


def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)


pred_subset_to_mpcat40 = np.array([
    0,  # void
    1,  # wall
    2,  # floor
    3,  # chair
    10, # sofa
    5,  # table
    11, # bed
    19, # stool
])


"""
Reads contact label names and color scheme of mpcat40 in [0,1] rgb 
Adapted from https://github.com/mohamedhassanmus/POSA/blob/main/src/viz_utils.py

Args:
    path: path to mpcat40.tsv; default "/mpcat40.tsv"
    
Returns:
    index lookup list of contact label names; index nmupy array of rgb colors

"""

# The read_mpcat40 function reads the TSV file containing the material categories and color codes. It extracts the label names and converts the hex color codes to RGB values in the range [0, 1]. The function returns two outputs: the index lookup list of contact label names and an index numpy array of RGB colors.

# By using this function, you can retrieve the label names and color codes for the material categories in the mpcat40 dataset, which can be used for visualization or any other application where material information is required.
def read_mpcat40(path="mpcat40.tsv"):
    import pandas as pd
    mpcat40 = pd.read_csv(path, sep='\t')
    label_names = list(mpcat40['mpcat40'])
    color_coding_hex = list(mpcat40['hex'])
    color_coding_rgb = []
    for hex_color in color_coding_hex:
        h = hex_color.lstrip('#')
        rgb = list(int(h[i:i + 2], 16) for i in (0, 2, 4))
        color_coding_rgb.append(rgb)
    color_coding_rgb = np.array(color_coding_rgb) / 255.0
    return label_names, color_coding_rgb


"""
Creates Open3D TriangleMesh object from vertices and faces numpy arrays

Args:
    vertices:   numpy array of vertices
    faces:      numpy array of faces

Returns:
    mesh:       Open3D TriangleMesh object
"""
def create_o3d_mesh_from_vertices_faces(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector([])
    mesh.vertex_normals = o3d.utility.Vector3dVector([])
    mesh.triangle_normals = o3d.utility.Vector3dVector([])
    return mesh


"""
Creates Open3D PointCloud object from points and optional color

Args:
    points:     numpy array of 3D coordinates
    colors:     numpy array of RGB colors; default: all white

Returns:
    pcd:        Open3D PointCloud object
"""
def create_o3d_pcd_from_points(points, colors=None):
    if colors is None:
        colors = np.ones_like(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


"""
Creates Open3D TriangleMesh object from vertices and faces numpy arrays

Args:
    vertices:   numpy array of vertices
    faces:      numpy array of faces

Returns:
    mesh:       Open3D TriangleMesh object
"""
def create_o3d_mesh_from_vertices_faces(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector([])
    mesh.vertex_normals = o3d.utility.Vector3dVector([])
    mesh.triangle_normals = o3d.utility.Vector3dVector([])
    return mesh


"""
Converts an Open3D TriangleMesh object to a trimesh object
This mesh only transfers vertices, faces, vertex colors,  vertex normals, face_normals

Args:
    o3d_mesh: Open3D TriangleMesh object
    
Returns:
    mesh: trimesh object
"""
def trimesh_from_o3d(o3d_mesh):
    vertices = np.array(o3d_mesh.vertices)
    faces = np.array(o3d_mesh.triangles)
    vertex_colors = np.array(o3d_mesh.vertex_colors)
    o3d_mesh.compute_vertex_normals()
    vertex_normals = np.array(o3d_mesh.vertex_normals)
    o3d_mesh.compute_triangle_normals()
    face_normals = np.array(o3d_mesh.triangle_normals)
    mesh = trimesh.Trimesh(
        vertices = vertices,
        faces = faces,
        face_normals = face_normals,
        vertex_normals = vertex_normals,
        vertex_colors = vertex_colors,
        process = False
    )
    return mesh


"""
Generates SDF from a trimesh object, and save relevant files to a given directory.

Args:
    mesh:                   input trimesh object
    dest_json_path:         output JSON path
    dest_sdf_path:          output SDF path
    dest_voxel_mesh_path:   output voxel mesh (obj) path; default "" and will not save voxel mesh
    grid_dim:               SDF grid_dim (N); default 256
    print_time              if True, prints time taken for generating SDF; default True
    
Returns:
    centroid:               centroid of the SDF
    extents:                extents of the SDF
    sdf:                    numy array of N x N x N containing SDF values 
"""
def generate_sdf(mesh, dest_json_path, dest_sdf_path, dest_voxel_mesh_path="", grid_dim=256, print_time=True):
    from mesh_to_sdf import mesh_to_voxels, sample_sdf_near_surface
    # Save centroid and extents data used for transforming vertices to [-1,1] while query
    # vertices = mesh.vertices - mesh.bounding_box.centroid
    # vertices *= 2 / np.max(mesh.bounding_box.extents)
    #Centroid: The centroid represents the geometric center of the mesh's bounding box. It is a point in 3D space that helps in positioning and orienting the mesh relative to other objects or coordinate systems. In the code provided, the centroid is saved as part of the SDF representation, allowing for proper transformation and alignment of vertices during later queries or computations.
    # The extents describe the size or dimensions of the mesh's bounding box. It represents the maximum extent in each dimension (width, height, and depth) of the mesh. Similar to the centroid, saving the extents is useful for transforming the vertices of the mesh to a standardized range, such as [-1, 1].
    centroid = mesh.bounding_box.centroid
    extents = mesh.bounding_box.extents
    # Save centroid and extents as SDF
    json_dict = {}
    json_dict['centroid'] = centroid.tolist()
    json_dict['extents'] = extents.tolist()
    json_dict['grid_dim'] = grid_dim
    json.dump(json_dict, open(dest_json_path, 'w'))
    
    if print_time:
        start_time = time.time() 
    
    sdf = mesh_to_voxels(mesh, voxel_resolution=grid_dim)
    
    if dest_voxel_mesh_path != "":
        import skimage.measure
        vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=0)
        voxel_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        voxel_mesh.export(open(dest_voxel_mesh_path, 'w'), file_type='obj')
    
    if print_time:
        print("Generating SDF took {0} seconds".format(time.time()-start_time))
    
    np.save(dest_sdf_path, sdf)
    
    centroid = np.copy(centroid)
    extents = np.copy(extents)
    
    return centroid, extents, sdf
    

"""
Creates per frame human mesh for sequence

Args:
    vertices_path: directory storing npy files of vertices
    faces_path:    path to faces obj file; default: POSA_dir/mesh_ds/mesh_2.obj
    
Returns:
    meshes: list of Open3D TriangleMesh representing human at each frame
"""
def read_sequence_human_mesh(vertices_path, faces_path=os.path.join("mesh_ds", "mesh_2.obj")):
    vertices = np.load(open(vertices_path, "rb"))
    faces = trimesh.load(faces_path, process=False).faces
    meshes = []
    for frame in range(vertices.shape[0]):
        meshes.append(create_o3d_mesh_from_vertices_faces(vertices[frame], faces))
    return meshes


"""
Merges a list of meshes into a single mesh

Args:
    meshes: a list of Open3D TriangleMesh
    
Returns:
    mesh: merged mesh
"""
def merge_meshes(meshes, skip_step=0):
    mesh_vertices = []
    mesh_faces = []
    num_verts_seen = 0
    if skip_step == 0:
        idxs = list(range(0, len(meshes)))
    else:
        idxs = list(range(0, len(meshes), skip_step))
    for i in idxs:
        mesh = meshes[i]
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.triangles)
        mesh_vertices.append(vertices)
        mesh_faces.append(faces + num_verts_seen)
        num_verts_seen += vertices.shape[0]
    mesh_vertices = np.concatenate(mesh_vertices, axis=0)
    mesh_faces = np.concatenate(mesh_faces, axis=0)
    return create_o3d_mesh_from_vertices_faces(mesh_vertices, mesh_faces)
    

"""
Given vertices and faces array, write mesh to path

Args:
    vertices:   mesh vertices as numpy array
    faces:      mesh faces as numpy array
    path:       output path
"""
def write_verts_faces_obj(vertices, faces, path):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(open(path, 'w'), file_type='obj')
    
    
    
"""
Given vertices and contact labels, estimate floor height

Args:
    vertices:           human mesh vertices as numpy array, shape: [frames, 655, 3]
    contact_labels:     contact labels for each vertex, shape: [frames, 655, 1]
    floor_offset:       floor distance from lowest cluster median, along negative up axis, in meters; default 0.0
"""
def estimate_floor_height(vertices, contact_labels, floor_offset=0.0):
    from sklearn.cluster import DBSCAN
    np.set_printoptions(threshold=np.inf) 
    print("Estimating height ", vertices.shape, contact_labels.shape)
    # print(contact_labels[0:20])
    #Estimating height  (256, 655, 3) (256, 655)
    floor_verts_heights = []
    for frame in range(contact_labels.shape[0]):
        floor_verts = vertices[frame, contact_labels[frame] == 2]
        if len(floor_verts) > 0:
            floor_verts_heights.append(floor_verts[:,2].min())
    
    floor_verts_heights = np.array(floor_verts_heights)
    # print("Shape of floor_verts_heights that is clustered", floor_verts_heights.shape)
    # (202, ) for MPH112...
    # DBSCAN (Density-Based Spatial Clustering of Applications with Noise). 
    clustering = DBSCAN(eps=0.005, min_samples=3).fit(np.expand_dims(floor_verts_heights, axis=1))    
    min_median = float('inf')
    all_labels = clustering.labels_
    for label in np.unique(all_labels):
        clustered_heights = floor_verts_heights[all_labels == label]
        median = np.median(clustered_heights)
        if median < min_median:
            min_median = median
    return min_median - floor_offset
    

"""
Given object vertices and faces, rotate object around X axis for 90 degrees, and move object such that lowest vertex has z = 0

Args:
    vertices:   object mesh vertices as numpy array
    faces:      object mesh faces as numpy array
    write_path: if not empty, write aligned object to given path; default ""
"""    
def align_obj_to_floor(verts, faces, write_path=""):
    from scipy.spatial.transform import Rotation as R
    # zyx is different from ZYX!!
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html
    # "Up to 3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for intrinsic rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations"
    r = R.from_euler('XYZ', np.asarray([90, 0, 0]), degrees=True) 
    aligned_verts = r.apply(verts) 
    min_z_val = aligned_verts[:, 2].min()
    height_trans = 0 - min_z_val  
    aligned_verts[:, 2] += height_trans
    if write_path:
        print("Writing floor aligned obj to", write_path)
        write_verts_faces_obj(aligned_verts, faces, write_path)
    return aligned_verts


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + (kmat @ kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def normalize_orientation(verts_can, associated_joints, device):
    """
    Compute a rotation about z-axis to make pose from the first frame facing directly out of the screen, then applies
    this rotation to poses from all the following frame.
    Parameters
    ----------
    verts_can: a sequence of canonical vertices.
    associate_joints: (Nverts, 1), tensor of associated joints for each vertex
    device: the device on which tensors reside
    """
    n_verts = verts_can.shape[1]
    first_frame = verts_can[0]
    joint1_indices = (associated_joints == 1)
    joint2_indices = (associated_joints == 2)
    verts_in_joint1 = first_frame[joint1_indices]
    verts_in_joint2 = first_frame[joint2_indices]
    joint1 = torch.mean(verts_in_joint1, dim=0).reshape(1, 3)
    joint2 = torch.mean(verts_in_joint2, dim=0).reshape(1, 3)
    orig_direction = (joint1 - joint2).squeeze().detach().cpu().numpy()
    orig_direction[2] = 0  # Project the original direction to the xy plane.
    dest_direction = np.array([1, 0, 0])
    rot_mat = rotation_matrix_from_vectors(orig_direction, dest_direction)
    rot_mat = torch.tensor(rot_mat, dtype=torch.float32).to(device)
    verts_can = torch.matmul(rot_mat, verts_can.permute(2, 0, 1).reshape(3, -1).to(device))
    verts_can = verts_can.reshape(3, -1, n_verts).permute(1, 2, 0)
    return verts_can.detach().cpu().numpy()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming you have the following variables defined:
# total_obj_list: List of bounding boxes
# scene_center: Numpy array representing the scene center

def fun(scene_center, total_obj_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the scene center
    ax.scatter(scene_center[0], scene_center[1], scene_center[2], c='red', label='Scene Center')

    # Plotting the bounding boxes and their half extents
    for bbox in total_obj_list:
        half_extent = bbox.get_half_extent()
        center = bbox.get_center()

        # Extracting the x, y, z coordinates from the center and half_extent arrays
        x, y, z = center[0], center[1], center[2]
        dx, dy, dz = half_extent[0], half_extent[1], half_extent[2]

        # Plotting the bounding box
        ax.add_artist(plt.Rectangle((x - dx, y - dy), 2 * dx, 2 * dy, facecolor='none', edgecolor='blue'))
        ax.plot([x - dx, x + dx], [y - dy, y - dy], [z - dz, z - dz], 'blue')
        ax.plot([x - dx, x + dx], [y + dy, y + dy], [z - dz, z - dz], 'blue')
        ax.plot([x - dx, x - dx], [y - dy, y + dy], [z - dz, z - dz], 'blue')
        ax.plot([x + dx, x + dx], [y - dy, y + dy], [z - dz, z - dz], 'blue')
        ax.plot([x - dx, x - dx], [y - dy, y - dy], [z + dz, z - dz], 'blue')
        ax.plot([x - dx, x + dx], [y + dy, y + dy], [z + dz, z - dz], 'blue')
        ax.plot([x - dx, x - dx], [y - dy, y + dy], [z + dz, z - dz], 'blue')
        ax.plot([x + dx, x + dx], [y - dy, y + dy], [z + dz, z - dz], 'blue')
        ax.plot([x - dx, x - dx], [y - dy, y - dy], [z + dz, z + dz], 'blue')
        ax.plot([x - dx, x + dx], [y + dy, y + dy], [z + dz, z + dz], 'blue')
        ax.plot([x - dx, x - dx], [y - dy, y + dy], [z + dz, z + dz], 'blue')
        ax.plot([x + dx, x + dx], [y - dy, y + dy], [z + dz, z + dz], 'blue')

        # Plotting the half extents as vectors
        ax.quiver(x, y, z, dx, dy, dz, color='green', label='Half Extent')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set legend
    ax.legend()

    # Show the plot
    plt.show()


