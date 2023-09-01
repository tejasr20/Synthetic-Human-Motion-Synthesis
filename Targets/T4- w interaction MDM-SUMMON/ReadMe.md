Once MDM is completely set up with smplx in it, 
modify new.yml and run 
"python -m transfer_model --exp-cfg config_files/new.yml" from smplx dir.
This will do whole step to convert to summon format.
(SMPL->SMPLX-> normalize-> downsample)
Then run contactFormer on output. 
 <!-- run "python merged.py" after changes to new.yml.  -->
