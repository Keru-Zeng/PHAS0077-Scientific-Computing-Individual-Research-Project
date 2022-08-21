# PHAS0077-Scientific-Computing-Individual-Research-Project

## Functions
The aim of the codes is build U-net based convolutional neural networks to segment different cells. The codes build two models: first is built by Abhinav Sagar called old model and second is built based on PLA data called new model.

## Usage
People should run train_model_new first and then run test_model_new to see segmented cells using new model.

People should run train_model_old first and then run test_model_old to see segmented cells using old model.

## Detail Explanation
stage1_train is the folder to train the model used by Abhinav Sagar. 

tifs_unanalysed is the folder to store some sample of PLA images and the train_set and test_set are construct from it. Moreover, masks in train_set are built by label-studio:
Maxim Tkachenko, Mikhail Malyuk, Andrey Holmanyuk and Nikolai Liubimov, "Label Studio: Data labeling software", 2020-2022, accessed from https://github.com/heartexlabs/label-studio.

premodel_functions contains some functions which are useful in writting the report such as how to transform PLA into 2D inputs and example_pre_model is an example to show the result of it, which will be shown in report.

model_functions contains some useful functions which will be used in training models.

label_cell_segmentation is Otsu method to segment the cells.

train_model_old and test_model_old are used to build old model and predict and segment cells using old model.

train_model_new and test_model_new are used to build new model and predict and segment cells using new model.

## Dependency

tensorflow                2.6.0

keras                     2.6.0

black                     22.6.0

scikit-image              0.19.2

scipy                     1.7.3

pims                      0.6.1

opencv-python             4.6.0.66

numpy                     1.21.5

matplotlib                3.5.2

tqdm                      4.64.0

## Other useful Dependency

h5py                      3.7.0

pyqt5                     5.15.7

## Reference
These codes are modified base on: 
1. Sagar, Abhinav. “Nucleus Segmentation using U-Net: How can deep learning be used for segmenting medical images?”, 2019. Accessed from https://towardsdatascience.com/nucleus-segmentation-using-u-net-eceb14a9ced4. 

2. "Label image regions", accessd from https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_label.html

3. Adrian Rosebrock. "Watershed OpenCV", 2015. Accessed from https://pyimagesearch.com/2015/11/02/watershed-opencv/
