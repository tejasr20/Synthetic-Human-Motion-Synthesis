#!/bin/bash

# Set variable names
name="Punches"
output_path="/data/tejasr20/summon/predictions/scene/$name/human/mesh"
fitting_results_path="/data/tejasr20/summon/predictions/scene/$name"
vertices_path="/data/tejasr20/summon/data/mdm/chair2/ext+rev/chair2_verts_can.npy" # can add anything here, will not be used 
path_to_obj_files="/data/tejasr20/motion-diffusion-model/save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000200000_seed10_the_person_walked_forward_and_punched_the_air/sample00_rep00_obj"
sl=1 # spare length 
ni=2 # num_iterations 
vis=false
generate_meshes=true

# Check if the meshes already exist
if [ -d "$output_path" ]; then
    echo "Rotated meshes already exist. Skipping data creation."
    generate_meshes=false
fi

# Perform data creation if needed
if [ "$generate_meshes" = true ]; then
    # Generate .obj meshes
    cd /data/tejasr20/summon/
    # conda activate summon
    python convert.py -if "$path_to_obj_files" -of "$output_path"
    echo "Rotated meshes saved to $output_path"
fi

# Run scene completion
python scene_completion.py --fitting_results_path "$fitting_results_path" \
    --path_to_model "atiss_ckpt" \
    --obj_dataset_path "3D_Future/models" \
    --spare_length "$sl" \
    --num_iter "$ni"

# Visualize fitting results if vis is true
if [ "$vis" = true ]; then
    conda activate summon
    python vis_fitting_results.py --fitting_results_path "$fitting_results_path" \
        --vertices_path "$vertices_path"
fi
