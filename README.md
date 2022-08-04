# PHAS0077-Scientific-Computing-Individual-Research-Project

## Reference

These codes are modified mainly base on 
@misc{sagarkaggle,
  Author = {Abhinav Sagar},
  Title = {Kaggle Solutions},
  Year = {2019},
  Journal = {Github},
}

Accessed from: https://towardsdatascience.com/nucleus-segmentation-using-u-net-eceb14a9ced4
## Functions
The aim of the codes is build U-net based convolutional neural networks to segment different cells. The codes build two models: first is built by Abhinav Sagar called old model and second is built based on PLA data.
## Usage
stage1_train is the folder to train the model used by Abhinav Sagar. 
tifs_unanalysed is the folder store the PLA images and the train_set and test_set are construct from it. Moreover, masks in train_set are built by label-studio:
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
