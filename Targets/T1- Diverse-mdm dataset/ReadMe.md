Including two files, one is
extract_humanML_texts.py: Provide input paths to this of humanML texts, Output folder, and output file path.
run using "python extract_humanML_texts.py"
The sentences extracted will be saved to output_folder path with same naming convention as HumanMl3d text.
(direct correspondence from 000001.txt ->000001.txt) 

Second file is HML.sh
To run 
use "chmod +x ./HML.sh" once, followed by "./HML.sh"
Before running, set the parameters for mdm_path, input_text_path, output_dir, hml3d_text_path, model_path
according to the comments given in the shell script.
If you want to limit iterations, i.e output folders/ videos, please set max_iteration to a positive integer 
of that number != -1. 
You will see output according to HumanML naming convention in output_dir. 
000000/ folder will have 000000.txt with dense text representation. 