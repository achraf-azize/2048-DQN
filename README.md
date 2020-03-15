# 2048

### Please read these instructions carefully

Here are the instructions in order to run the code. The code is divided into two independant parts: a Jupyter Notebook (2048_DQN.ipynb) and 3 python files (main.py, game_display.py, classes.py). Additionnally, the 3 files parameters_reward are the parameters of the trained DQNs associated to each of the 3 rewards considered. These files can be used for the reproduction of the results. 

First download the whole repository in a Zip file.

## Notebook

Please open the notebook in a Google Colab environment: https://colab.research.google.com/notebooks/intro.ipynb, then import the notebook. For that, you will need a google account. Indeed, we use Google Colab in order to increase the speed of our programs, thanks to Cuda (using GPUs).

Before running anything, upload the 3 .dms files (on the left, click on the folder and import the files). Also, modify the runtime type (runtime, modify, select GPUs).

Run all the cells of the first section (main functions and classes), and do not run those in the section Training agents for different rewards: they will not work without specific requirements, so they are present only to display the code and the outputs found in the report. We created specific sections that can enable to reproduce the results (1 section for training, the other for testing). 
In those sections, you can change parameters such as the reward and the number of epochs.


## Python files

The python files provide the opportunity to display a specific game played by one of the 3 agents considered. Put all the files (3 .dms files included) in the same project, and run the main.py file (you can change the reward parameter to change the considered reward).
