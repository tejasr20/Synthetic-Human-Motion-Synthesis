To first generate the required human motion output from MDM, with object meshes
: can run the two commands in MDM repo or simply run "./gen.sh" after running chmod +x. 
Preferably place gen.sh in MDM folder and script.sh in summon folder. 
Activate the two environments at the appropriate time. 
Change the arguments as fit. 
Now we need to convert to point cloud, rotate, and apply scene completion. 
This can be done by running the shell script "./script.sh"
Please fill in the path of the output obj files of step 1 appripriately(And others)
The two hyperparameters are sl and ni, scene length and number of iterations (or 
objects to be added to the scene): modify as you see fit. 

