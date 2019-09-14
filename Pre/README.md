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
> [[Raport]](https://drive.google.com/file/d/1qWt1FUVE5lxVT9z9ZGO4IziYSVd_o72T/view?usp=sharing)

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
<img width="900" src="plots/Results_for_all_models_train_val_test.png">
</p>
<p align="justify">

Red line - our baseline LSTM encoder decoder PR model, the worst result; Light green line - the best result at the moment; Strong green line - second result.


<p align="center">
<img width="900" src="plots/Testting_results_for_all_models_PR.png">
</p>
<p align="justify">

Testing results for all models. Denormalized MSE for pitch and roll at 10s in predicted sequences. Red line - our baseline and the worst result; Light green line - the best result at the moment; Strong green line - second result.


<p align="center">
<img width="900" src="plots/TRAIN_VAL_all_models_loss_function.png">
</p>
<p align="justify">

<p align="center">
<img width="900" src="plots/TRAIN_TEST_all_models_loss_function.png">
</p>
<p align="justify">

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

`autoencoder_train.py`: used to train autoencoder model

`test.py`: used to predict all results

`utils.py`: some useful functions

`hyperband.py`: implementation of the Hyperband algorithm

`get_hyperparameters_configuration.py`: define Hyperband space

`earlyStopping.py`: implementation of the Early Stoping technique

`help_plot.py, help_plot_2.py, help_plot_3.py `:  scripts to display some useful charts

`plot_compare_predicted_and_original_PR.py`: script to plot original and predicted pitch and roll

`plot_evolution_PR_over_predicted_seq.py`: script to plot evolution of predicted pitch and roll over sequence


# Step guidance:

1. clone repository

2. create directory tree:

---> Pre

------> 3dmodel

------> results

3. download the images dataset [here](https://drive.google.com/drive/folders/1RF8_wFfcIM0GIklXflPYv-tK3uaEWSSZ?usp=sharing) if you have not yet and put the dataset under the directory **3dmodel**

4. for train, goto Pre's parent folder and run command:

```
python3 -m Pre.train --train_folder Pre/3dmodel/test_4_episode_ --num_epochs 50 --batchsize 24 --learning_rate 0.001 --opt "adam" --seed 42 --no_cuda True --model_type "LSTM_encoder_decoder_PR" --encoder_latent_vector 300 --decoder_latent_vector 300 --future_window_size 20 --past_window_size 20 --frame_interval 12 --weight_decay 0.001 --use_n_episodes 540 --change_fps False --test 0
```

 
5. for prediction, goto Pre's parent folder and run command:
```
python3 -m Pre.test -f Pre/3dmodel/test_4_episode_ --num_epochs 50 --batchsize 24 --learning_rate 0.001 --opt "adam" --seed 42 --no_cuda True --load_weight_file "Pre/results/train_CNN_LSTM_encoder_decoder_images_PR_using_20_s_to_predict_30_s_lr_0.0001937_2019-08-12 18_29_35/weight/CNN_LSTM_encoder_decoder_images_PR_predict_30_s_using_20_s_lr_0.0001937_tmp.pth" --model_type "LSTM_encoder_decoder_PR" --encoder_latent_vector 300 --decoder_latent_vector 300 --future_window_size 20 --past_window_size 20 --frame_interval 12 --weight_decay 0.001 --use_n_episodes 540 --change_fps False
```
options:
 - train_folder    (str): folder's prefix where dataset is stored (path + episodes) - [Pre/3dmodel/test_4_episode_ ]
 - num_epochs      (int): number of epochs - [50]
 - batchsize       (int): batchsize - [32]
 - opt             (str): optimizer type  - ['adam', 'sgd']
 - learning_rate   (float): learning_rate - [0.000000001 - 0.01]
 - seed            (int): number to fix random processes - [42]
 - cuda            (boolean): True if we can use GPU
 - load_weight     (boolean): True if we will load model
 - load_weight_date(str): date of the test (part of the path)
 - model_type      (str): model type  - ['CNN_stack_FC_first', 'CNN_stack_FC', 'CNN_LSTM_image_encoder_PR_encoder_decoder', 'CNN_PR_FC', 'CNN_LSTM_encoder_decoder_images', 'LSTM_encoder_decoder_PR', 'CNN_stack_PR_FC', 'CNN_LSTM_encoder_decoder_images_PR', 'CNN_LSTM_decoder_images_PR'] 
 - encoder_latent_vector (int): size of encoder latent vector - [0 - 10000]
 - decoder_latent_vector (int): size of decoder latent vector - [0 - 10000]
 - future_window_size    (int): number of seconds to predict - [0 - 30]
 - past_window_size      (int): number of seconds using like input - [0 - 30]
 - frame_interval        (int): interval at witch the data was generated - [12 if 2 fps]
 - weight_decay          (float): L2 penalty - [0.000000001 - 0.01]
 - use_n_episodes        (int): number of episodes use for work -  [0 - 540]
 - test_dir              (str): if you run a parameter test, all results will be stored in test folder
 - change_fps            (boolean): True if we want to use 1 fps when data was generated with 2 fps.
 - test                  (int): - [0 - train model ; 1 - hyperband test (hyperparameters search)]
 
# Some issues

1. Be careful when setting parameters, check constants: for example, the sequence time [LEN_SEQ] should be large enough to include past window size + future window size. To set it go to constants.py file!
