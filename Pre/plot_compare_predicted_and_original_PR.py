"""
this code is used to plot figure comparing the original boat parameters and predictions ones
"""

import matplotlib
import matplotlib.pyplot as plt
import json
import argparse
from Pre.constants import RES_DIR
import numpy as np


def denormalization(x, min_v = -90.0, max_v = 90.0 ):
    return (x+1)*(max_v-min_v)/2 + min_v
# Pre/results/test_CNN_LSTM_encoder_decoder_images_PR_using_20_s_to_predict_30_s_lr_0.0001115124882_2019-08-21 15:11:31/labels/originCNN_LSTM_encoder_decoder_images_PR_use_20_s_to_predict_1:30_lr_0.0001115124882.json
# Pre/results/test_CNN_LSTM_encoder_decoder_images_PR_using_20_s_to_predict_30_s_lr_0.0001115124882_2019-08-21 15:11:31/labels/predCNN_LSTM_encoder_decoder_images_PR_use_20_s_to_predict_1:30_lr_0.0001115124882.json
# train_CNN_LSTM_encoder_decoder_images_PR_using_20_s_to_predict_30_s_lr_0.0001937_2019-08-12 18_29_35
parser = argparse.ArgumentParser(description='Test a line detector')
parser.add_argument('-o', '--origin_file', help="Original file. Ex: Pre/results/train_LSTM_encoder_decoder_PR_using_10_s_to_predict_12_s_lr_0.0001_2019-08-24 22:40:34/labels/originLSTM_encoder_decoder_PR_use_10_s_to_predict_20:24_lr_0.0001.json", type=str, required=True)
parser.add_argument('-p', '--prediction_file', help="Predictin file. Ex: Pre/results/train_LSTM_encoder_decoder_PR_using_10_s_to_predict_12_s_lr_0.0001_2019-08-24 22:40:34/labels/predLSTM_encoder_decoder_PR_use_10_s_to_predict_20:24_lr_0.0001.json", type=str, required=True)
args = parser.parse_args()


norm = True
# read data
originData = args.origin_file
predData = args.prediction_file
labelsOrigin = json.load(open(originData))
labelsPred = json.load(open(predData))
frames1 = list(labelsOrigin.keys())
frames2 = list(labelsPred.keys())

# sort keys
frames1.sort(key=lambda name: int(name.strip().replace('"',"")))
frames2.sort(key=lambda name: int(name.strip().replace('"',"")))

# obtain roll and pitch lists respectively
if norm:
    originRoll = [denormalization(labelsOrigin[str(key)][0]) for key in frames1]
    originPitch = [denormalization(labelsOrigin[str(key)][1]) for key in frames1]
    predRoll = [denormalization(labelsPred[str(key)][0]) for key in frames2]
    predPitch = [denormalization(labelsPred[str(key)][1]) for key in frames2]
else:
    originRoll = [labelsOrigin[str(key)][0] for key in frames1]
    originPitch = [labelsOrigin[str(key)][1] for key in frames1]
    predRoll = [labelsPred[str(key)][0] for key in frames2]
    predPitch = [labelsPred[str(key)][1] for key in frames2]


# frames: str -> int
frames1 = [int(key) for key in frames1]
frames2 = [int(key) for key in frames2]
# calculate MAE
rollMAE=0
pitchMAE=0
idx = 0
for i in frames2:
    rollMAE += abs(originRoll[idx]-predRoll[idx])
    pitchMAE += abs(originPitch[idx]-predPitch[idx])
    idx += 1
print('MAE of roll is', rollMAE/len(frames2))
print('MAE of pitch is', pitchMAE/len(frames2))

# calculate MSE
rollMSE=0
pitchMSE=0
idx = 0
for i in frames2:
    rollMSE += abs((originRoll[idx]-predRoll[idx])**2)
    pitchMSE += abs((originPitch[idx]-predPitch[idx])**2)
    idx += 1
print('MSE of roll is', rollMSE/len(frames2))
print('MSE of pitch is', pitchMSE/len(frames2))

# calculate number of local extremum
roll_max_local=0
for u in range (1,len(originRoll)-1):
    if ((originRoll[u]>originRoll[u-1])&(originRoll[u]>originRoll[u+1])):
        roll_max_local=roll_max_local+1
print('roll peak time:', roll_max_local)
pitch_max_local=0
for u in range (1,len(originPitch)-1):
    if ((originPitch[u]>originPitch[u-1])&(originPitch[u]>originPitch[u+1])):
        pitch_max_local=pitch_max_local+1
print('pitch peak time:', pitch_max_local)


# plot roll
plt.figure(1)
# resize pic to show details
plt.figure(figsize=(30, 12))
plt.plot(frames1, originRoll, 'r-', label='original roll')
plt.plot(frames2, predRoll, 'b-', label='predicted roll')
plt.title("roll - frame", fontsize = 18)
plt.xlabel("Frames", fontsize = 18)
plt.ylabel("Roll", fontsize = 18)
plt.legend(loc='upper right', prop={'size': 16})
plt.show()

# plot pitch
plt.figure(2)
# resize pic to show details
plt.figure(figsize=(30, 12))
plt.plot(frames1, originPitch, 'r-', label='original pitch')
plt.plot(frames2, predPitch, 'b-', label='predicted pitch')
plt.title("pitch - frame", fontsize = 18)
plt.xlabel("Frames", fontsize = 18)
plt.ylabel("Pitch", fontsize = 18)
plt.legend(loc='upper right', prop={'size': 16})
plt.show()
