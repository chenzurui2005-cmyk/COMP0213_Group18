# COMP0213_Group18

The objective of this code is generating and evaluating grasp candidates using machine learning (ML) classification using pybullet and then use Logistic Regression to evaluate on it.

# How to install dependencies
Clone the entire project, then run "pip install -r requirements.txt" in your python environment(FOr example, conda.)

# How to run the thing
Run Models.py to see the results for the logistic regression models on the pre-generated size-120 dataset.
If you want to use another set of data, run Data_gen.py . It will prompt you to enter the desired gripper type and object type, then wait for the simulation to finish. The new datafile will replace the old one.
If you want to change the dataset size, go to Data_gen.py and change variable "GRASP_VALID_TOTAL".
