# merges the output of the main transfer_model script

import torch
import numpy as np
from pathlib import Path
import pickle
from scipy.spatial.transform import Rotation as R

KEYS = [
"transl",
"global_orient",
"body_pose",
"betas",
"left_hand_pose",
"right_hand_pose",
"jaw_pose",
"leye_pose",
"reye_pose",
"expression",
"vertices",
"joints",
"full_pose",
"v_shaped",
"faces"
]

IGNORED_KEYS = [
"vertices",
"faces",
"v_shaped"
]

def aggregate_rotmats(x):
    x = torch.cat(x, dim=0).detach().cpu().numpy()
    s = x.shape[:-2]
    x = R.from_matrix(x.reshape(-1, 3, 3)).as_rotvec()
    x = x.reshape(s[0], -1)
    return x

aggregate_function = {k: lambda x: torch.cat(x, 0).detach().cpu().numpy() for k in KEYS}
aggregate_function["betas"] = lambda x: torch.cat(x, 0).mean(0).detach().cpu().numpy()

for k in ["global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose", "full_pose"]:
    aggregate_function[k] = aggregate_rotmats

def merge(output_dir, gender, data_list):
    output_dir = Path(output_dir)
    assert output_dir.exists()
    assert output_dir.is_dir()
	# if(!output_dir.exists()):
    #  os.makedirs(output_dir, exist_ok=True)
    # get list of all pkl files in output_dir with fixed length numeral names
    # pkl_files = [f for f in output_dir.glob("*.pkl") if f.stem != "merged"]
    # pkl_files = [f for f in sorted(pkl_files, key=lambda x: int(x.stem))]
    # assert "merged.pkl" not in [f.name for f in pkl_files]

    merged = {}
    new= {}
    # iterate over keys and put all values in lists
    keys = set(KEYS) - set(IGNORED_KEYS)
    for k in keys:
        merged[k] = []
    for data in data_list:
         for k in keys:
            if k in data:
                merged[k].append(data[k])
    b = torch.cat(merged["betas"], 0)
    print("betas:")
    for mu, sigma in zip(b.mean(0), b.std(0)):
        print("  {:.3f} +/- {:.3f}".format(mu, sigma))

    # aggregate all values
    for k in keys:
        merged[k] = aggregate_function[k](merged[k])
	# reye_pose (2, 1, 3)
	# leye_pose (2, 1, 3) : combine those to for pose_eye in pickle_amass
	# left_hand_pose (2, 45)
 	# right_hand_pose (2, 45):  combine those to for pose_hand in pickle_amass
	# transl (2, 3) : same as transl in pickle_amass
	# expression (2, 10): :not there
	# betas (16,): same 
	# global_orient (2, 3) : same as root_orient in pickle_amass
	# full_pose (2, 165) ->corresponds to poses in pickle_amass
	# body_pose (2, 63): same as pose_body in pickle amass 
	# jaw_pose (2, 3): same as pose_jaw in pickle_amass
	
	# joints (2, 144, 3)
    # add gender
    merged["gender"] = gender
    for key in merged:
        if(key!="gender"):
            print(key, merged[key].shape)
    # for key in merged:
    #     if(key!="gender"):
    #         print(key, type(merged[key]))
        # print(key, )
        
    # blah= np.array(merged)
    # print(blah.shape)//
    # save merged data to same output_dir
    # with open(output_dir / "merged.pkl", "wb") as f:
    #     pickle.dump(merged, f)
    
    d= {"poses":"full_pose", "root_orient":"global_orient", "trans":"transl", "pose_body":"body_pose", "betas":"betas", "gender":"gender", "reye_pose": "reye_pose", "leye_pose": "leye_pose", "left_hand_pose":"left_hand_pose", "right_hand_pose": "right_hand_pose", "expression": "expression" }
    for key in d:
        new[key]= merged[d[key]]
    
    return new 
    
    #  # Save merged data to same output_dir as npz file
    # npz_output_path = output_dir / "merged.npz"
    # np.savez(npz_output_path, **merged)
    # print("Saved merged data as .npz file:", npz_output_path)

	# # Save merged data to same output_dir as npz file
    # npz_output_path = output_dir / "merged_new.npz"
    # np.savez(npz_output_path, **new)
    # print("Saved merged data with modified keys as .npz file:", npz_output_path)

    # # Save merged data to same output_dir as npy file
    # npy_output_path = output_dir / "merged.npy"
    # np.save(npy_output_path, merged)
    # print("Saved merged data as .npy file:", npy_output_path)
    # Save merged data to same output_dir as npy file
    # npy_output_path = output_dir / "merged.npy"
    # np.save(npy_output_path, merged)
    # print("Saved merged data as .npy file:", npy_output_path)

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(description='Merge output of transfer_model script')
#     parser.add_argument('output_dir', type=str, help='output directory of transfer_model script')
#     parser.add_argument('--gender', type=str, choices=['male', 'female', 'neutral'], help='gender of actor in motion sequence')
#     args = parser.parse_args()
#     merge(args.output_dir, args.gender)

