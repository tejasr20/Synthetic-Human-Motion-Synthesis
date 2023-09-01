import os
import open3d as o3d
import numpy as np
import re

ply_folder = "/data/tejasr20/summon/predictions/scene/chair3_euler2/human/mesh"  # Path to the folder containing .ply files
output_folder = "/data/tejasr20/summon/data/mdm/chair3/NEW/vertices_can/"  # Path to the output folder for modified meshes
name = "euler3"  # Name of the output mesh
save_rotated = True  # Flag to save rotated meshes in output folder
mesh_output_folder = "/data/tejasr20/summon/predictions/scene/jacks/human/mesh"  # Path to the folder for saving the rotated meshes

# List all .ply files in the folder
ply_files = [f for f in os.listdir(ply_folder) if f.endswith(".ply")]

# Sort the .ply files based on the filenames
ply_files.sort(key=lambda x: int(re.search(r"\d+", x).group()))

# List to store the vertices
vertices_list = []

# Create the output mesh folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create the rotated mesh folder if save_rotated is True
if save_rotated and not os.path.exists(mesh_output_folder):
    os.makedirs(mesh_output_folder)

# Iterate over each .ply file
for ply_file in ply_files:
    print(ply_file)
    # Load the .ply file as an Open3D mesh object
    mesh = o3d.io.read_triangle_mesh(os.path.join(ply_folder, ply_file))
    # mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, np.pi/2, np.pi/2)), center=(0, 0, 0))
    # mesh.rotate(mesh.get_rotation_matrix_from_xyz((0, np.pi/2, np.pi)), center=(0, 0, 0))
    mesh.rotate(mesh.get_rotation_matrix_from_xyz((np.pi/2,0, 0)),
              center=(0, 0, 0))
    # Convert mesh vertices to a NumPy array
    vertices = np.asarray(mesh.vertices)

    # Append the vertices to the list
    vertices_list.append(vertices)

    if save_rotated:
        # Save the rotated mesh in the output folder with the same name as the input mesh
        output_file = ply_file
        o3d.io.write_triangle_mesh(os.path.join(mesh_output_folder, output_file), mesh)
        print(f"Rotated mesh {output_file} saved.")

# Convert vertices_list to a NumPy array
vertices_array = np.array(vertices_list)

# # Save the vertices array as an .npy file
output_file = name + "_verts_can.npy"
np.save(os.path.join(output_folder, output_file), vertices_array)

# # Load the original vertices from an existing .npy file
# original_vertices = np.load("/data/tejasr20/summon/data/mdm/chair2/vertices_can/chair2_verts_can.npy")  # Replace with the path to your original vertices .npy file
# print(original_vertices.shape)

# # Compare the computed vertices with the original vertices
# if np.array_equal(vertices_array, original_vertices):
#     print("The computed vertices are equal to the original vertices.")
# else:
#     print("The computed vertices are NOT equal to the original vertices.")
