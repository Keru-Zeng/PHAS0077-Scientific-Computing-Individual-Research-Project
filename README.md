# PHAS0077-Scientific-Computing-Individual-Research-Project

## Functions
The aim of the codes is build U-net based convolutional neural networks to segment different cells. The codes build two models: first is built by Abhinav Sagar called old model and second is built based on PLA data called new model.

## Usage
People should run train_model_new first and then run test_model_new to see segmented cells using new model.

People should run train_model_old first and then run test_model_old to see segmented cells using old model.

## Detail Explanation
stage1_train is the folder to train the model used by Abhinav Sagar. 

tifs_unanalysed is the folder store PLA images and the train_set and test_set are construct from it. Moreover, masks in train_set are built by label-studio:
@misc{Label Studio,
  title={{Label Studio}: Data labeling software},
  url={https://github.com/heartexlabs/label-studio},
  note={Open source software available from https://github.com/heartexlabs/label-studio},
  author={
    Maxim Tkachenko and
    Mikhail Malyuk and
    Andrey Holmanyuk and
    Nikolai Liubimov},
  year={2020-2022},
}

premodel_functions contains some functions which are useful in writting report and example_pre_model is an example to show the result in it.

model_functions contains useful functions which will be used in training model.

label_cell_segmentation is another segment method which could be regarded as traditional method apart from U-net.

train_model_old and test_model_old are used to build old model and predict and segment cells using old model.

train_model_new and test_model_new are used to build new model and predict and segment cells using new model.

## Reference
These codes are modified base on 
@misc{sagarkaggle,
  Author = {Abhinav Sagar},
  Title = {Kaggle Solutions},
  Year = {2019},
  Journal = {Github},
}
,which could be accessed from: https://towardsdatascience.com/nucleus-segmentation-using-u-net-eceb14a9ced4

and "Label image regions", which could be accessd from https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_label.html
