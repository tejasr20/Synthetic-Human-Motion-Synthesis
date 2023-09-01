import numpy as np

def load_and_visualize_array(path):
    # Load the array from the specified path
    arr = np.load(path,  allow_pickle=True)

    # Print the shape of the array
    print("Array shape:", arr.shape)

    # Print the first few rows of the array
    print("First few rows:")
    print(arr[:5])  # Adjust the slice to print more or fewer rows

# Path to the numpy array file
array_path = '/data/tejasr20/motion-diffusion-model/save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000200000_seed10_The_person_kicked_with_his_left_foot_and_then_walked_straight_in_his_living_room/sample00_rep00_smpl_params.npy'

# Call the function to load and visualize the array
load_and_visualize_array(array_path)
