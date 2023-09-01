Add fit_mult_best_obj.py to summon folder in summon environment. 
Run using the same command as fit_best_object.py but with some changes:
1) Make sure that you passed the --save_probability flag when generating contacts. 
2) Use the --input_probability flag with the command
3) There are two new variables, change and orient. 
If you wish to change the scene by using new, semantically similar objects , pass "--change 1"
If you wish to change the scene by using a sub-optimal fit of the best objects, 
pass "--orient 1".
An example of the same, 
"python fit_mult_best_obj.py --sequence_name MPH11_00150_01 --vertices_path data/proxd_valid/vertices/MPH11_00150_01_verts.npy --contact_labels_path predictions/proxd_valid/MPH11_00150_01.npy --output_dir fitting_results --input_probability --change 1"

