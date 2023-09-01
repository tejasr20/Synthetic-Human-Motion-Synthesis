Repository for summer internship 23/24 in Samsung Korea based on Synthetic Human Motion Synthesis. 
Derived from [MDM]([url](https://github.com/GuyTevet/motion-diffusion-model)https://github.com/GuyTevet/motion-diffusion-model) and [SUMMON]([url](https://github.com/onestarYX/summon)https://github.com/onestarYX/summon).
MDM is used to generate accurate human motion in SMPL body mesh form without scene.
SUMMON can be used for adding scene auto regressively to SMPL-X motions from the PROX-D Dataset, for example. 
1) Papers- Contains the two research papers for motion diffusion model and Summon.
2) Photos- Some output photos
3) Presentation- The final presentation
4) Summon- Contains the summon repository with some additions
5) Targets- Contains four folders, T1, T2, T3, T4 with instructions to run all of them inside their respective readme. T1 is diverse MDM dataset, which can be used to generate a diverse large dataset
6) by extracting text prompts from HumanML3d and aplying that to MDM. T2 is Diverse Scene generation, which means generating diverse scenes from a single input human motion by manipulating probabilities
7) of ContactFormer(Refer to presentation for details). T3 MDM-ATISS is a way of generating human motion with scene but without interaction of human with surrounding using MDM. T4 involves integration
8) of SMPL- SMPL-X conversion from MDM output to a single step instead of 6-7 steps using the SMPL-X transfer model within MDM. This could be used to generate scene for arbitrary input human
9) motion, but generates incorrect positions of chair possibly due to incorrect conversion or lack of generalizability of contactFormer. 
10) Videos- Some example output videos: You can see examples of videos generated from MDM that have had scene added to them by employing a conversion of MDM output to SMPL body mesh and then a rotation to correct orientation, fillowed by ATISS.
11) You also see examples of diverse scene generation, the same human motion with multiple scenes, a chair replaced with a sofa, or another type of chair from the 3D Future object dataset. These are among the targets explained
12) in the targets folder. 
