# PHAS0077-Scientific-Computing-Individual-Research-Project

## Functions
The aim of the codes is to build a U-net based convolutional neural networks to identify the location and shapes of cells' nuclei and segment different cells. The codes train the model using two datasets: first is a large, publicly available dataset in the folder called stage1_train and second is the PLA dataset in the folder called train_set. Besides, the dataset named test_set will be used to test the model and predict the model. Finally, by using Otsu method with the binary pixel's Euclidean distance and the watershed method, we could see the results of the segmented cells.

## Usage
Since there are two datasets, we should run the model twice to see the results of each situations.

1. People should run train_model_new first and then run test_model_new to see the results of the prediction of the model trained by PLA dataset and the results of segmented cells.

2. People should run train_model_old first and then run test_model_old to see the results of the prediction of the model trained by stage1_train folder and the results of segmented cells.

## Results
The results will be shown in two folders in each situation, and if we want to start a new dataset, don't forget to clear all of the previous models' results. sep_set contains the results of segmented cells and pred_set contains the prediction of the model.

## Detail Explanation

tifs_unanalysed is the folder to store some raw sample of PLA images and the test_set are constructed from it. 

masks in train_set are built by label-studio.

premodel_functions contains some functions which are useful in writting the report such as how to transform raw PLA data into 2D inputs and example_pre_model is an example results, which will be shown in the report.

model_functions contains some useful functions which will be used in training the model such as applying data augmentation and building U-net model.

train_model_old is to use stage1_train to train the U-net model.
test_model_old is used to test the model and predict the model using test_set and shows the segmented cells.

train_model_new is to use train_set to train the U-net model.
test_model_new is used to test the model and predict the model using test_set and shows the segmented cells.

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

2. Adrian Rosebrock. "Watershed OpenCV", 2015. Accessed from https://pyimagesearch.com/2015/11/02/watershed-opencv/

3. Maxim Tkachenko, Mikhail Malyuk, Andrey Holmanyuk and Nikolai Liubimov, "Label Studio: Data labeling software", 2020-2022, accessed from https://github.com/heartexlabs/label-studio.