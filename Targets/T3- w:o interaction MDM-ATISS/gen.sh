#!/bin/bash

# Navigate to the motion-diffusion-model directory
cd /data/tejasr20/motion-diffusion-model/
text="the person walked forward and punched the air."
underscored_text="${text// /_}"  # Replaces spaces with underscores

# Remove the full stop at the end if present
last_char="${underscored_text: -1}"  # Get the last character
if [ "$last_char" = "." ]; then
    underscored_text="${underscored_text%?}"  # Remove the last character
fi

path="/data/tejasr20/motion-diffusion-model/save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000200000"
model_path="/data/tejasr20/motion-diffusion-model/save/humanml_trans_enc_512/model000200000.pt"
# python -m sample.generate --model_path "$model_path"  --text_prompt "$text"

# Activate the environment
# conda activate mdm1 # please keep activated

# Read input_path and name from config file or set them here
# input_path="/data/tejasr20/motion-diffusion-model/save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000200000_seed10_A_person_is_skipping_rope/sample00_rep00.mp4"
seed="10"
sample="00"
rep="00"
input_path="${path}_seed${seed}_${underscored_text}/sample${sample}_rep${rep}.mp4"
echo "${input_path}"
name="skips"

# Generate .obj meshes
generate_meshes=true

# # Check if the meshes already exist
# if [ -d "/data/tejasr20/summon/scene/$name/human/mesh" ]; then
#     echo "Rotated meshes already exist. Skipping data creation."
#     generate_meshes=false
# fi

# Perform data creation if needed
if [ "$generate_meshes" = true ]; then
    # Generate .obj meshes
    python -m visualize.render_mesh --input_path "$input_path"
    echo "Generated object meshes"
fi
/data/tejasr20/motion-diffusion-model/save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000200000_seed10_the_person_walked_forward_and_punched_the_air/results.npy