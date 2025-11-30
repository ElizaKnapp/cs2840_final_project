### Instructions
1. Download the data bundle zip file here: https://drive.google.com/file/d/1N7ycuPqcMOn331JNeFIZnI7Xx8r0-lcM/view?usp=sharing
2. Replace the data folder of this project with the data inside of that folder
3. Type `make` into the command line

The make command does the following
1. Prepares the data for OT (make prepare)
2. Runs hybrid ground metric OT (make sweep). This returns the optimal alpha for the data clustering that you have selected.
3. Runs the experiment to evaluate the OT. You must select an alpha to evaluate. (make evaluate)

You can select which OT data to prepare (how we define the clusters) by changing the environment.yaml file. 
You can also select the size of the OT clusters by changing the environment.yaml file.
