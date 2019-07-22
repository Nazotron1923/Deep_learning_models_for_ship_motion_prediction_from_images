"""
this code is used to plot figure comparing the original boat parameters and predictions ones
"""
# run this code under ssh mode, you need to add the following two lines codes.
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import json
import argparse
from Pre.constants import RES_DIR
import numpy as np
from math import factorial

def denormalization(x, min_v = -90.0, max_v = 90.0 ):
    return (x+1)*(max_v-min_v)/2 + min_v

parser = argparse.ArgumentParser(description='Test a line detector')
parser.add_argument('-t', '--time_gap', help='How many seconds you want to predict', default=10, type=int)
parser.add_argument('-u', '--use_sec', help='How many seconds use for prediction', default=10, type=int)
parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=0.0003, type=float)
parser.add_argument('--model_type', help='Model type: cnn', default="CNN_LSTM_encoder_decoder_images_PR", type=str, choices=['CNN_stack_FC_first', 'CNN_stack_FC', 'CNN_LSTM_image_encoder_PR_encoder_decoder', 'CNN_PR_FC', 'CNN_LSTM_encoder_decoder_images', 'LSTM_encoder_decoder_PR', 'CNN_stack_PR_FC', 'CNN_LSTM_encoder_decoder_images_PR', 'CNN_LSTM_decoder_images_PR'])
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


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


norm = True
frame_interval = 12
predict_n_im = int(24/frame_interval)*args.time_gap
future_frames = np.array([i for i in range(1, predict_n_im+1) ])
future_roll_MSE_for_frames = np.zeros(predict_n_im)
future_pitch_MSE_for_frames = np.zeros(predict_n_im)

for i_frame in range(predict_n_im):
    print("----------------------------{}-------------------------------".format(i_frame))
    tmp_origin_i = args.origin_file + "train_{}_using_{}_s_to_predict_{}_s_lr_{}_{}/labels/origin{}_use_{}_s_to_predict_{}:{}_lr_{}.json".format(args.model_type, args.use_sec, args.time_gap, args.learning_rate, args.date, args.model_type, args.use_sec, future_frames[i_frame], predict_n_im, args.learning_rate )
    tmp_pred_i = args.prediction_file + "train_{}_using_{}_s_to_predict_{}_s_lr_{}_{}/labels/pred{}_use_{}_s_to_predict_{}:{}_lr_{}.json".format(args.model_type, args.use_sec, args.time_gap, args.learning_rate, args.date, args.model_type, args.use_sec, future_frames[i_frame], predict_n_im, args.learning_rate )

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
plt.figure(1)
# resize pic to show details
plt.figure(figsize=(30, 12))
plt.plot(future_frames, future_roll_MSE_for_frames, 'r-', label='roll MSE ')
plt.plot(future_frames, future_pitch_MSE_for_frames, 'b-', label='pitch MSE')
plt.plot(future_frames, tt, 'g--', label='Result')
plt.plot(future_frames, avg_MSE, 'y--', label='avg MSE ')

plt.title("MSE - frame")
plt.xlabel("Frames")
plt.ylabel("MSE")
plt.grid(True)
plt.legend(loc='upper right')


tmp_im = "train_{}_using_{}_s_to_predict_{}_s_lr_{}_{}".format(args.model_type, args.use_sec, args.time_gap, args.learning_rate, args.date)
str_t = str(args.prediction_file+tmp_im+'/MSE_evolution_denorm.png')
plt.savefig(str_t)
