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

Project data & results: [[Google drive]](https://drive.google.com/drive/folders/1RF8_wFfcIM0GIklXflPYv-tK3uaEWSSZ?usp=sharing)

### Blender simulation

As mentioned above the 3D graphics generator Blender was used to generate dataset. Blender is the free and open source 3D creation suite. It supports the entirety of the 3D pipeline—modeling, rigging, animation, simulation, rendering, compositing and motion tracking, even video editing and 2D design.

Thanks to the work presented in [raport](https://github.com/manubatet/Ship-simulator/blob/master/ENSTA_ShipSimulator.pdf), we can use ready-made scripts to generate the necessary data. All needed is to specify the necessary parameters of the ocean, a camera position and an image expansion. (See how to install blender in [repository](https://github.com/Nazotron1923/ship-ocean_simulation_BLENDER))

### Datasets

In general, we generated 540 episodes (400 images per episode) = 216,000 images = 1800 minutes = 30 hours of simulations.

Also, to save memory, we chose rendering 2 frames per second. This is the best option so that the differences between two frames are noticeable and they do not "duplicate" the information.

<p align="center">
  <img width="600" src="plots/gen_img.jpg">
</p>
<p align="justify">

Project data: [[Google drive]](https://drive.google.com/drive/folders/1RF8_wFfcIM0GIklXflPYv-tK3uaEWSSZ?usp=sharing)

### Data preprocessing

Before using the generated data, it needs to be processed. Since we used an artificial generator, all images have the same size and scaling or cropping is not required. However, data needs to be normalized, and so, every pixel value is converted into values between -1 and 1 (see Formula 1).
Since we are going to use pitch and roll values for some models, we normalize it with absolute values. The absolute values of these angles (pitch and roll) are -90 and 90 degrees, and so the Formula 2 is used to normalize the data in [-1, 1]:

<p align="center">
  <img width="500" src="plots/preprocessing.png">
</p>
<p align="justify">

### Models

To solve our pitch and roll prediction problem, existing model architectures will be used, whose advantages will be combined to achieve the best result. To begin, we consider the basic architectures for images - Convolutional Neural Networks and time series (in our case the simulation of the sea surface) - Long Short-Term Memory networks.

<p align="center">
  <img width="600" src="plots/numenclature.PNG">
</p>
<p align="justify">

9 models were created:
- CNN stack FC model
  -- version 1: predict only one pitch and roll
  -- version 2: predict sequence of pitch and roll
- CNN stack PR FC model   
- CNN PR FC model
- LSTM encoder decoder PR model
- CNN LSTM img-encoder PR-encoder decoder model
- CNN LSTM encoder decoder images PR model
- CNN LSTM encoder decoder images model
- CNN LSTM images PR model


CNN stack version 1 FC  |  CNN stack version 2 FC
:-------------------------:|:-------------------------:
<img src="plots/CNN_stack_FC_first.png" width="486" />  | <img src="plots/CNN_stack_FC.png" width="486" />

CNN stack PR FC model  |  CNN PR FC model
:-------------------------:|:-------------------------:
<img src="plots/CNN_stack_PR_FC.png" width="486" />  | <img src="plots/CNN_PR_FC.png" width="486" />

              LSTM encoder decoder PR model
<p align="center">
  <img width="600" src="plots/LSTM_encoder_decoder_PR.png">
</p>
<p align="justify">

CNN LSTM img-encoder PR-encoder decoder model  |  CNN LSTM encoder decoder images PR model
:-------------------------:|:-------------------------:
<img src="plots/CNN_LSTM_im_encoder_pr_encoder_decoder.png" width="486" />  | <img src="plots/CNN_LSTM_encoder_decoder_images_PR.png" width="486" />

CNN LSTM encoder decoder images model  |  CNN LSTM images PR model
:-------------------------:|:-------------------------:
<img src="plots/CNN_LSTM_encoder_decoder_images.png" width="486" />  | <img src="plots/CNN_LSTM_images_PR.png" width="486" />



### Results

First, the basic settings were tested;

<p align="center">
<img width="300" src="plots/param_for_test.png">
</p>
<p align="justify">

The results of the experiments can be found in Table, which shows the normalized average MSE of the sum of pitch and roll over the predicted sequence (of 24 frames of length).

<p align="center">
<img width="900" src="plots/TRAIN_VAL_all_models_loss_function.png">
</p>
<p align="justify">

Red line - our baseline LSTM encoder decoder PR model, the worst result; Light green line - the best result at the moment; Strong green line - second result.


<p align="center">
<img width="900" src="plots/TRAIN_TEST_all_models_loss_function.png">
</p>
<p align="justify">

Testing results for all models. Denormalized MSE for pitch and roll at 10s in predicted sequences. Red line - our baseline and the worst result; Light green line - the best result at the moment; Strong green line - second result.

# The best configuration

Using Hyperband [algorithm](https://github.com/zygmuntz/hyperband) the best configuration for CNN_LSTM_encoder_decoder_images_PR model was found:

<p align="center">
<img width="300" src="plots/best_config.png">
</p>
<p align="justify">


LSTM encoder decoder pitch [baseline]  |  LSTM encoder decoder roll [baseline]
:-------------------------:|:-------------------------:
<img src="plots/baseline_pitch_10s_v1.png" width="486" />  | <img src="plots/baseline_roll_10s_v1.png" width="486" />


CNN LSTM encoder decoder images PR model pitch at 15 sec  |  CNN LSTM encoder decoder images PR model roll at 15 sec
:-------------------------:|:-------------------------:
<img src="plots/CNN_LSTM_encoder_decoder_images_PR_pitch_15s_best_1fps.png" width="486" />  | <img src="plots/CNN_LSTM_encoder_decoder_images_PR_roll_15s_best_1fps.png" width="486" />

CNN LSTM encoder decoder images PR model pitch at 30 sec |  CNN LSTM encoder decoder images PR model roll at 30 sec
:-------------------------:|:-------------------------:
<img src="plots/CNN_LSTM_encoder_decoder_images_PR_pitch_30s_best_1fps.png" width="486" />  | <img src="plots/CNN_LSTM_encoder_decoder_images_PR_roll_30s_best_1fps.png" width="486" />


### Conclusion

In this work, in order to predict ship motion from images, nine deep neural network models were created and tested. Such variety of models is caused by the complexity of the problem. Empirical results show that models with LSTM parts and using additional information such as the current ship motion improve the pitch and roll prediction accuracy. The best result was achieved by a [CNN LSTM encoder-decoder images PR] model. The best combination of parameters was found using Hyperband algorithm (learning rate, weight decay, encoder latent vector size and decoder latent vector size). In general, the model normally exhibits fluctuations, and sometimes skips large peaks (not being accurately enough for the angle value). The problem is still not completely solved and can have many improvements. Even in the best version of the created model there are problems such as overfitting, poor generalization, etc.
Still, not solved the problem with the data, to achieve a better result and good working model real data from the ship is needed.
However, reasonable predictions are achieved with the proposed model.

### License

This project is released under a [GPLv3 license](LICENSE).

### Dependencies

To run all scripts the presented environment is needed:

 - environment.yml


# Files explanations


`constants`: defines some main constants of the project

`models.py`: neural network models

`train.py`: used to train all models

`test.py`: used to predict all results


# Step guidance:

1. download the code and add environment path by changing the code in `autoRun.sh`

2. download the images dataset [here](https://drive.google.com/drive/folders/1RF8_wFfcIM0GIklXflPYv-tK3uaEWSSZ?usp=sharing) if you have not yet and put the dataset under the directory **3dmodel**

3. for train, goto Pre's parent folder and run command:
```
python3 -m Pre.train -tf Pre/3dmodel/test_4_episode_
```

4. for prediction, goto Pre's parent folder and run command:
```
python3 -m Pre.test -f Pre/3dmodel/test_4_episode_
```

# Some issues

1. Be careful when setting parameters, check constants: for example, the sequence time [LEN_SEQ] should be large enough to include past window size + future window size.
