git clone https://github.com/onestarYX/summon
#create environment as asked in the repository. 

cd contactFormer
python predict_contact.py ../data/amass --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --output_dir predictions/contact/amass/

# This will create contact predictions at "predictions/contact/"
# can change as you see fit 

I have also attached a shell script, amass.sh

Add to directory summon/

First chmod +x amass.sh
Then to run ./amass.sh

If visualization doesnt work simpply run the script again by commenting fit_best_obj. 

Change the name as you see fit in the file. 
It will take the contact points, use it to fit best object, 
and show you the visualization. can comment fit_best obj if already done
You will see that chair is at incorrect position for 13_03...npy 
Change the folder_name to Prox and the name accordingly 
to see the results for prox. 
An exmple name for amass is "13_03_stageii",
for prox is "MPH11_00150_01" 


Inaccurate results for AMASS. Similar issues faced when using MDM output. 
(makes sense because it is trained on HumanML3d which is derived from AMASS)

the code for how they have processed the dataset will be in contactFormer/gen_dataset.py
They call a function pkl_to_canonical()(in the file data_utils.py) and process there.
Also note the function normalize_orientation() which is called before predicting 
contact points in predict_contact.py. I have also used this.  

In pickle_amass_vertices.py you will see a modified version of that, looping over 
an npz instead of pickle files. Of the four lists produced, the one that is used 
is all_vertices_can_ds2. 

https://github.com/mohamedhassanmus/POSA : at the bottom of this repository is what
I believe this is the format required. The processing for the same is done in 
data_utils.py in POSA(with a slightly different pkl_to_canonical function )->
I tried taking inspiration from here to fix orientation. 
Actually the multiplication with euler angles rotation is equivalent to 
using rotate_mesh on an open3d mesh which I had already tried before. 
I think the correct rotation for mdm output is 3pi/2, pi, pi about x,y,z respectively. 
ALterntatively use normalize_orientation func 

I have included some files I have created for the same. (rotate_mesh.py can rotate by an angle you desire, or simply add the same line in vis_fitting_results.py before rendering)
The pickle_amass_vertices.py I have attached has the code that I created for the 
last step after converting mdm output to smplx. I have tried many combinations. 

script.sh contains the steps for scene completion from mdm. 