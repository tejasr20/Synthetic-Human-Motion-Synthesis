#!/bin/bash

# Change directory
# cd /data/tejasr20/summon/

# Set the name variable
name="75_17_stageii"
folder_name="amass" # for amass.
# folder_name="proxd_valid/vertices_can" # for prox 

# # Set the visualization variable (0 = no visualization, 1 = visualization)
# vis=1

# # Run the first command
# python fit_best_obj.py --sequence_name "$name" --vertices_path "./data/${folder_name}/${name}_verts_can.npy" --contact_labels_path "predictions/contact/${folder_name}/${name}.npy" --output_dir "predictions/scene/${folder_name}"

# Run the second command with visualization option
# ep=1
# python vis_fitting_results.py --fitting_results_path "./predictions/scene/amass/${name}" --vertices_path "/data/tejasr20/summon/data/amass/${name}_verts_can.npy" --empty "$ep" 
python vis_fitting_results.py --fitting_results_path "./predictions/scene/${folder_name}/${name}" --vertices_path "/data/tejasr20/summon/data/${folder_name}/${name}_verts_can.npy" 
