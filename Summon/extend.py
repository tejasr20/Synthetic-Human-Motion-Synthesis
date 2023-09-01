import os
import numpy as np

def double_vector_size(npy_path, output_path, reverse=False):
    # Load the npy file
    vector = np.load(npy_path)

    # Get the shape of the vector
    shape = vector.shape

    # Double the size of the vector by copying the last element x times
    last_element = vector[-1]
    # Reverse the first x elements
    reversed_elements = np.flip(vector[:shape[0]], axis=0)
    if(reverse):
        # Double the size of the vector by concatenating the reversed elements
        doubled_vector = np.concatenate([vector, reversed_elements])
    else:
        doubled_vector = np.concatenate([vector, np.tile(last_element, (shape[0], 1, 1))])
        
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the modified vector as a new npy file
    np.save(output_path, doubled_vector)

    # Print the new size
    new_size = doubled_vector.shape[0]
    print(f"New vector size: {new_size}")

# Example usage
npy_path = '/data/tejasr20/summon/data/mdm/chair2/extended/chair2_verts_can.npy'  # Replace with the path to your input npy file
output_path = '/data/tejasr20/summon/data/mdm/chair2/ext+rev/chair2_verts_can.npy'  # Replace with the desired output path

double_vector_size(npy_path, output_path, True)
