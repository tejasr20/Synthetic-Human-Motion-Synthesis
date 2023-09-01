# #!/bin/bash

# # Set the input path and variable name
# input_path="/data/tejasr20/summon/data/proxd_train/vertices_can"
# variable_name="proxd_train"

# # Change directory to contactFormer
# cd /data/tejasr20/summon/contactFormer

# # Run predict_contact.py
# python predict_contact.py "${input_path}" --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --output_dir "/data/tejasr20/summon/predictions/contact/${variable_name}" --save_probability

# # Move back to the parent directory
# cd ..

# # Loop through .npy files in the input path
# for file in "${input_path}"/*.npy; do
#     if [[ "${file}" =~ (.*)_verts_can\.npy$ ]]; then
#         # Extract the name (x) from the file name
#         x="${BASH_REMATCH[1]}"
# 		variable_name="proxd_train"
        
#         # Run fit_best_obj.py
#         python fit_mult_best_obj.py --sequence_name "${x}" --vertices_path "${file}" --contact_labels_path "/data/tejasr20/summon/predictions/contact/${variable_name}/${x}.npy" --output_dir fitting_results/${variable_name}/unchanged/ --input_probability

#         # Set vis variable to 0
#         vis=1

# 		if [ "${vis}" -eq 1 ]; then
#             # Run vis_fitting_results.py if vis is 1
#             python vis_fitting_results.py --fitting_results_path "fitting_results/${variable_name}/unchanged/${x}" --vertices_path "${file}"
#         fi
# 		change=1
# 		if [ "${change}" -eq 1 ]; then
#             # run again 
#            python fit_mult_best_obj.py --sequence_name "${x}" --vertices_path "${file}" --contact_labels_path "/data/tejasr20/summon/predictions/contact/${variable_name}/${x}.npy" --output_dir fitting_results/${variable_name}/changed/ --input_probability --change
# 		   if [ "${vis}" -eq 1 ]; then
# 				# Run vis_fitting_results.py if vis is 1
# 				python vis_fitting_results.py --fitting_results_path "fitting_results/${variable_name}/changed/${x}" --vertices_path "${file}"
#            fi
#         fi

#     fi
# done
       #!/bin/bash

# Set the input path and variable name
input_path="/data/tejasr20/summon/data/proxd_train/vertices_can"
variable_name="proxd_train"

# Change directory to contactFormer
cd /data/tejasr20/summon/contactFormer

# Run predict_contact.py
python predict_contact.py "${input_path}" --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --output_dir "/data/tejasr20/summon/predictions/contact/${variable_name}" --save_probability

# Move back to the parent directory
cd ..

# Loop through .npy files in the input path
for file in "${input_path}"/*.npy; do
    if [[ "${file}" =~ (.*)_verts_can\.npy$ ]]; then
        # Extract the name (x) from the file name
        x="${BASH_REMATCH[1]}"
        variable_name="proxd_train"
        
        # Get the full path of the input file
        vertices_path="${input_path}/${x}_verts_can.npy"

        # Run fit_best_obj.py
        python fit_mult_best_obj.py --sequence_name "${x}" --vertices_path "${vertices_path}" --contact_labels_path "/data/tejasr20/summon/predictions/contact/${variable_name}/${x}.npy" --output_dir fitting_results/${variable_name}/unchanged/ --input_probability

        # Set vis variable to 0
        vis=1

        if [ "${vis}" -eq 1 ]; then
            # Run vis_fitting_results.py if vis is 1
            python vis_fitting_results.py --fitting_results_path "fitting_results/${variable_name}/unchanged/${x}" --vertices_path "${vertices_path}"
        fi
        
        change=1
        if [ "${change}" -eq 1 ]; then
            # Run fit_mult_best_obj.py again with the --change argument
            python fit_mult_best_obj.py --sequence_name "${x}" --vertices_path "${vertices_path}" --contact_labels_path "/data/tejasr20/summon/predictions/contact/${variable_name}/${x}.npy" --output_dir fitting_results/${variable_name}/changed/ --input_probability --change

            if [ "${vis}" -eq 1 ]; then
                # Run vis_fitting_results.py if vis is 1
                python vis_fitting_results.py --fitting_results_path "fitting_results/${variable_name}/changed/${x}" --vertices_path "${vertices_path}"
            fi
        fi

    fi
done
