#!/bin/bash

# Navigate to the motion-diffusion-model directory
cd /data/tejasr20/motion-diffusion-model/
text="the person walked forward and punched someone."
model_path="data/tejasr20/motion-diffusion-model/save/humanml_trans_enc_512/model000200000.pt"
python -m sample.generate --model_path "${model_path}"  --text_prompt "${text}"

# Activate the environment
# conda activate mdm1 # please keep activated

# Read input_path and name from config file or set them here
# input_path="/data/tejasr20/motion-diffusion-model/save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000200000_seed10_A_person_is_skipping_rope/sample00_rep00.mp4"
seed="10"
sample="00"
rep="00"
input_path="${model_path}_seed${seed}_sample${sample}_rep${rep}.mp4"
name="skips"

# Generate .obj meshes
generate_meshes=true

# Check if the meshes already exist
if [ -d "/data/tejasr20/summon/scene/$name/human/mesh" ]; then
    echo "Rotated meshes already exist. Skipping data creation."
    generate_meshes=false
fi

# Perform data creation if needed
if [ "$generate_meshes" = true ]; then
    # Generate .obj meshes
    python -m visualize.render_mesh --input_path "$input_path"
	echo "Generated object meshes"
fi

