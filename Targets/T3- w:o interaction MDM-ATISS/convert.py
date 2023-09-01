import argparse
import os
import open3d as o3d
import functools as fc
import gen_human_meshes 
import numpy as np 

def convert_obj_to_ply(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    obj_files = [f for f in os.listdir(input_folder) if f.endswith('.obj')]
    obj_files = sorted(obj_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0][5:]))
    ind=0
    for obj_file in obj_files:
        obj_path = os.path.join(input_folder, obj_file)
        filename= "human_"+ str(int(os.path.splitext(os.path.basename(obj_file))[0][5:]))
        # Construct the output .ply file path
        ply_path = os.path.join(output_folder, filename + ".ply")
        ind+= 1
        # ply_path = os.path.join(output_folder, obj_file.replace('.obj', '.ply'))

        mesh = o3d.io.read_triangle_mesh(obj_path)
        mesh.rotate(mesh.get_rotation_matrix_from_xyz((3*np.pi/2, 2*np.pi/2, 2*np.pi/2)),
              center=(0, 0, 0))
        o3d.io.write_triangle_mesh(ply_path, mesh)

    print("Conversion completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert .obj files to .ply files')
    # parser.add_argument('-if', 'input_folder', help='Path to the input folder containing .obj files')
    # parser.add_argument('-of', 'output_folder', help='Path to the output folder to save the converted .ply files')
    parser.add_argument('-if', '--input_folder', metavar='input_folder',
                        help='Input folder containing .obj files')
    parser.add_argument('-of', '--output_folder', metavar='output_folder',
                        help='Output folder for storing .ply files')
    args = parser.parse_args()

    convert_obj_to_ply(args.input_folder, args.output_folder)
