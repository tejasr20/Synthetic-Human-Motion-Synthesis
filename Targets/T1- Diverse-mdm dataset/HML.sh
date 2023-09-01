#!/bin/bash

mdm_path="/data/tejasr20/motion-diffusion-model" # path to MDM
input_text_path="/data/tejasr20/motion-diffusion-model/dataset/HumanML3D/smpl_texts" # path to simplified texts
output_dir="./save/HML_mdm"
hml3d_text_path="/data/tejasr20/motion-diffusion-model/dataset/HumanML3D/texts" # path to original texts
model_path="./save/humanml_trans_enc_512/model000200000.pt"
max_iterations=-1  # Maximum number of iterations
# If max iterations is -1 then does not exit
# Change directory to the motion-diffusion-model path
cd "$mdm_path"

# Create the output directory
mkdir -p "$output_dir"

# Initialize counter variable
iterations=0

# Loop over each file in the input_text_path folder
for file_path in "$input_text_path"/*.txt; do
    # Check if maximum iterations reached and break if not -1
    if [ "$iterations" -ge "$max_iterations" ] && [ "$max_iterations" -ne -1 ]; then
        echo "Maximum iterations reached. Exiting loop."
        break
    fi
    
    # Increment counter
    ((iterations++))
    
    # Extract the file name without extension
    file_name=$(basename "$file_path" .txt)
    
    # Read the sentence x from the file
    x=$(head -n 1 "$file_path")
    
    # Create the directory for this iteration
    iteration_dir="$output_dir/$file_name"
    mkdir -p "$iteration_dir"
    
    echo "Processing file: $file_name"
    echo "Sentence: $x"
    echo "Output directory: $iteration_dir"
    
    # Call the python script to generate output
    python -m sample.generate --model_path "$model_path" --text_prompt "$x" --output_dir "$iteration_dir"
    
    # Copy the corresponding HML3d text file
    cp "$hml3d_text_path/$file_name.txt" "$iteration_dir"
    
    echo "Processing completed for file: $file_name"
done

# Exit the loop
