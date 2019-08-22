# Pre Project: Deep learning for predicting ship motion from images
***Author:*** [Nazar-Mykola KAMINSKYI](https://github.com/Nazotron1923)


Internship summer 2019


### Abstract

Today, artificial intelligence penetrates into all areas of human activity. Developed methods and recent advances in machine learning show good results and will gradually replace other \textit{"traditional"} approaches. AI allows to automate processes thereby improving efficiency and accuracy of tasks in co-operation with people. I worked on the problem of predicting ship motion from images of the sea surface.

First of all, using 3D graphics generator Blender, I created a dataset, which simulates the ship's movement through sea waves. This software also gives information about parameters of the boat (pitch and roll in our case). Data streams were structured in sequences and normalized for further processing. Then 9 different neural networks were developed and tested.
As I worked with time-ordered image sequences, I decided to use convolutional neural networks (CNN) for processing images and long short-term memory (LSTM) networks for time series processing.
Finally, I run the hyperband algorithm to find the best model hyperparameters and then all experiment results were analyzed. Also, some possible improvements are suggested.

Keywords: Computer Vision, Deep learning, ship motion, pitch and roll prediction, image processing, Blender, CNN, LSTM, Hyperband

> Deep learning for predicting ship motion from images.
> Nazar-Mykola KAMINSKYI, 2019.
> [[Raport]](https://drive.google.com/file/d/1-wrGLbEWiXemHF5n54mouZVaL28BICS_/view?usp=sharing)


### Blender simulation

TEXT + link MANU

### Datasets
TEXT
3D visualization of one of these graphs.
<p align="center">
  <img width="600" src="plots/gen_img.jpg">
</p>
<p align="justify">

Project data: [[Google drive]](https://drive.google.com/drive/folders/1RF8_wFfcIM0GIklXflPYv-tK3uaEWSSZ?usp=sharing)

### Data preprocessing

TEXT

### Models

TEXT

### Results

TEXT


# Acknowledgement
I received a lot of help from M. David Filliat and M. [Antonin RAFFIN](https://github.com/araffin). This work could not be done without their support. Merci beaucoup!

# License

This project is released under a [GPLv3 license](LICENSE).

# Dependencies

 - environment.yml


# Files explanations

`autoRun.sh`: runs all the codes automatically and generates the results. Usage: `bash autoRun.sh` or `sudo bash autoRun.sh` if your python modules is installed with sudo command.

`comparePare.xls`: a xls file which can calculate the parameters of a cnn network if you want to change some of parameters

`constants`: defines some main constants of the project

`models.py`: neural network models

`pltDiff.py`: used to plot figure comparing the original boat parameters and predictions ones

`pltModelTimegap.py`: read the result.txt and plot the model-timeGap-loss figures

`render.py`: used for render images data set with blender file. Run command:
```
blender model.blend --background --python render.py
```
or copy the code to blender's text editor and press "run script"

`result.txt`: contains the results of training (architectures of the network and test loss etc.). Generated automatically by `train.py`

`train.py`: used to train the model

`test.py`: used to predict the results

`transformData.py`: transform the data obtained from `blender.py` to the one which can be used for training

`insertKeyframe.py`: insert the prediction parameters into blender file `model.blend` and visualize the prediction in order to compare with the original ones. Usage: copy the code to blender's text editor and press "run script"

`model.blend`: 3d simulation file, use to generate images dataset

`labels.json`: ground truth data of boat's parameters

# Step guidance:

1. download the code and add environment path by changing the code in `autoRun.sh`

2. download the images dataset [here](https://drive.google.com/file/d/1wf86xezQeI804QpEtDDfxNovNXhN4Y6m/view?usp=sharing) if you have not yet and put the dataset under the directory **3dmodel** (frame 1--5000 are of object "boat", frame 5000--10000 are of object "boat1"), make sure the dataset folder's name is **mixData** and has file **labels.json** inside it. The prediction frame gap is 25 by default.

3. for train, goto Pre's parent folder and run command:
```
python3 -m Pre.train -tf Pre/3dmodel/mixData
```

4. for prediction, goto Pre's parent folder and run command:
```
python3 -m Pre.test -f Pre/3dmodel/mixData
```

5. or you can run autoRun.sh directly

# Some issues

1. After insert keyframes into blender model, the motion postures of the two boats are opposite
```
Make sure that the local coordinates of the two boats are the same, just change them into the same if not (rotate the prediction one)
```
