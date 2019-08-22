"""
this code is used to plot figure of evolution pitch and roll over predicted sequence
"""

import matplotlib
import matplotlib.pyplot as plt
import json
import argparse
from Pre.constants import RES_DIR
import numpy as np

def denormalization(x, min_v = -90.0, max_v = 90.0 ):
    return (x+1)*(max_v-min_v)/2 + min_v

# train_CNN_LSTM_encoder_decoder_images_PR_using_20_s_to_predict_30_s_lr_0.0001937_2019-08-12 18_29_35
parser = argparse.ArgumentParser(description='Test a line detector')
parser.add_argument('-t', '--time_gap', help='How many seconds you want to predict', default=30, type=int)
parser.add_argument('-u', '--use_sec', help='How many seconds use for prediction', default=20, type=int)
parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=0.0001937, type=float)
parser.add_argument('--model_type', help='Model type', default="CNN_LSTM_encoder_decoder_images_PR", type=str, choices=['CNN_stack_FC_first', 'CNN_stack_FC', 'CNN_LSTM_image_encoder_PR_encoder_decoder', 'CNN_PR_FC', 'CNN_LSTM_encoder_decoder_images', 'LSTM_encoder_decoder_PR', 'CNN_stack_PR_FC', 'CNN_LSTM_encoder_decoder_images_PR', 'CNN_LSTM_decoder_images_PR'])
parser.add_argument('-d', '--date', help='date of experiment', type=str, required=True)
parser.add_argument('-o', '--origin_file', help='Original file', default="Pre/results/", type=str)
parser.add_argument('-p', '--prediction_file', help='Predictin file', default="Pre/results/", type=str)
args = parser.parse_args()


def MAE(originRoll, predRoll, originPitch, predPitch, N):
    rollMAE=0
    pitchMAE=0
    idx = 0
    for i in range(N):
        rollMAE += abs(originRoll[idx]-predRoll[idx])
        pitchMAE += abs(originPitch[idx]-predPitch[idx])
        idx += 1

    return rollMAE/N, pitchMAE/N


def MSE(originRoll, predRoll, originPitch, predPitch, N):
    # calculate MSE
    rollMSE=0
    pitchMSE=0
    idx = 0
    for i in range(N):
        rollMSE += abs((originRoll[idx]-predRoll[idx])**2)
        pitchMSE += abs((originPitch[idx]-predPitch[idx])**2)
        idx += 1

    print('MSE of roll is', rollMSE/N)
    print('MSE of pitch is', pitchMSE/N)
    return rollMSE/N, pitchMSE/N


# If true data will be  denormalize
norm = False
frame_interval = 12
# For basic settings: if data genereted and tested with the same fps
predict_n_im = int(24/frame_interval)*args.time_gap
# If it is castom use next line
# predict_n_im = args.time_gap

future_frames = np.array([i for i in range(1, predict_n_im+1) ])
future_roll_MSE_for_frames = np.zeros(predict_n_im)
future_pitch_MSE_for_frames = np.zeros(predict_n_im)

for i_frame in range(predict_n_im):
    print("----------------------------{}-------------------------------".format(i_frame))
    tmp_origin_i = args.origin_file + "train_{}_using_{}_s_to_predict_{}_s_lr_{}_{}/labels/origin{}_use_{}_s_to_predict_{}_{}_lr_{}.json".format(args.model_type, args.use_sec, args.time_gap, args.learning_rate, args.date, args.model_type, args.use_sec, future_frames[i_frame], predict_n_im, args.learning_rate )
    tmp_pred_i = args.prediction_file + "train_{}_using_{}_s_to_predict_{}_s_lr_{}_{}/labels/pred{}_use_{}_s_to_predict_{}_{}_lr_{}.json".format(args.model_type, args.use_sec, args.time_gap, args.learning_rate, args.date, args.model_type, args.use_sec, future_frames[i_frame], predict_n_im, args.learning_rate )

    labelsOrigin = json.load(open(tmp_origin_i))
    labelsPred = json.load(open(tmp_pred_i))
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

    frames1 = [int(key) for key in frames1]
    frames2 = [int(key) for key in frames2]

    future_roll_MSE_for_frames[i_frame], future_pitch_MSE_for_frames[i_frame] = MSE(originRoll, predRoll, originPitch, predPitch, len(frames1))

avg_MSE = (future_roll_MSE_for_frames + future_pitch_MSE_for_frames)

avg_res = np.sum(future_roll_MSE_for_frames + future_pitch_MSE_for_frames)/(len(avg_MSE))
print("avg_res - ", avg_res)
tt = np.array([avg_res for i in range(len(avg_MSE))])

font = {'size'   : 22}

matplotlib.rc('font', **font)

plt.figure(1)
# resize pic to show details
plt.figure(figsize=(30, 14))
plt.plot(future_frames, future_roll_MSE_for_frames, 'r-', label='roll MSE ',  linewidth=4)
plt.plot(future_frames, future_pitch_MSE_for_frames, 'b-', label='pitch MSE',  linewidth=4)
plt.plot(future_frames, avg_MSE, 'g--', label='roll MSE + pitch MSE ',  linewidth=4)
plt.plot(future_frames, tt, 'k--', label='Average over future sequence (roll MSE + pitch MSE)',  linewidth=4)
print('future_roll_MSE_for_frames - ', future_roll_MSE_for_frames)
print("future_pitch_MSE_for_frames - ", future_pitch_MSE_for_frames)
print('tt - ', tt)
print("avg_MSE - ", avg_MSE)

plt.title("Test data\nMSE  - Predicted frame [1 fps]", fontsize = 26)
plt.xlabel("Predicted frame [1 fps] ", fontsize = 24)
plt.ylabel("MSE - loss function", fontsize = 24)
plt.grid(True)
plt.legend(loc='upper center', prop={'size': 28}, ncol=4,
            bbox_to_anchor=(0.5, -0.06),
          fancybox=True, shadow=True)


tmp_im = "train_{}_using_{}_s_to_predict_{}_s_lr_{}_{}".format(args.model_type, args.use_sec, args.time_gap, args.learning_rate, args.date)
str_t = str(args.prediction_file+tmp_im+'/MSE_evolution_norm_v2.png')
plt.savefig(str_t)
