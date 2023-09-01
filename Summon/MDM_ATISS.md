Please first generate the required MDM output using the command 
"python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --text_prompt "the person walked forward and is picking up his toolbox." or with any 
text prompt you desire that may be natural without scene. Then generate the meshes
using command given in MDM
Else directly run "gen.sh" using "./gen.sh" after changing the arguments appropriately.
You will get output motion with output .obj files in output_path. 

Activate the summon environment. 
Run the command ./script.sh that is in the summon folder after changing arguments.
This will create rotated converted human meshes and apply scene completion.
Please change arguments as you see fit.
To visualize the generated scene, you can set the vis variable to 1 in the script. 
Change sl and ni as required, to add more objects or spare length. 