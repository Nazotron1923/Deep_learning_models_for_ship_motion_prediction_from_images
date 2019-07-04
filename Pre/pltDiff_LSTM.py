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
parser.add_argument('-m', '--model', help='Model type', default="cnn", type=str, choices=['cnn','CNN_RNN','CNN_LSTM'])
parser.add_argument('-t', '--time_gap', help='TIME GAP', default=10, type=int)
parser.add_argument('-ni', '--num_images', help='Number of images to predict', default=10, type=int)
parser.add_argument('-o', '--origin_file', help='Original file', default="Pre/results/train_CNN_LSTM_encoder_decoder_images_PR_predict_5s_using_5s_2019-07-03 21:54:10/labels/CNN_LSTM_encoder_decoder_images_PR_origin_label_use_5_to_predict_2:10_lr_0.0001_.json", type=str)
parser.add_argument('-p', '--prediction_file', help='Predictin file', default="Pre/results/train_CNN_LSTM_encoder_decoder_images_PR_predict_5s_using_5s_2019-07-03 21:54:10/labels/CNN_LSTM_encoder_decoder_images_PR_pred_label_use_5_to_predict_2:10_lr_0.0001_.json", type=str)
args = parser.parse_args()

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
print('MAE error of pitch is', pitchMAE/len(frames2))

# calculate MAPE
rollMAPE=0
pitchMAPE=0
idx = 0
# for i in frames2:
    # rollMAPE += abs((labelsOrigin[str(i)][0]-predRoll[idx])/labelsOrigin[str(i)][0])
    # pitchMAPE += abs((labelsOrigin[str(i)][1]-predPitch[idx])/labelsOrigin[str(i)][1])
    # idx += 1
# print('MAPE of roll is', rollMAPE/len(frames2))
# print('MAPE of pitch is', pitchMAPE/len(frames2))

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


# Smoothing(optional)
# predRoll = savitzky_golay(np.asarray(predRoll), 101, 5)
# predPitch = savitzky_golay(np.asarray(predPitch), 101, 5)
# originRoll = savitzky_golay(np.asarray(originRoll), 101, 5)
# originPitch = savitzky_golay(np.asarray(originPitch), 101, 5)

# plot roll
plt.figure(1)
# resize pic to show details
plt.figure(figsize=(30, 12))
plt.plot(frames1, originRoll, 'r-', label='original roll')
plt.plot(frames2, predRoll, 'b-', label='prediction roll')
plt.title("roll - frame")
plt.xlabel("Frames")
plt.ylabel("Roll")
plt.legend(loc='upper right')
# plt.savefig(RES_DIR+args.model+'_KAMINSKYI_plt_test_'+str(args.time_gap)+'s_'+str(args.num_images)+'im_roll_final'+'.png')
str_t = str(RES_DIR+args.model+'_KAMINSKYI_plt_test_LSTM_'+str(args.time_gap)+'s_'+str(args.num_images)+':10_im_to_pred_roll_final'+'.png')
plt.savefig(str_t)
plt.show()

# plot pitch
plt.figure(2)
# resize pic to show details
plt.figure(figsize=(30, 12))
plt.plot(frames1, originPitch, 'r-', label='original pitch')
plt.plot(frames2, predPitch, 'b-', label='prediction pitch')
plt.title("pitch - frame")
plt.xlabel("Frames")
plt.ylabel("Pitch")
plt.legend(loc='upper right')
plt.savefig(RES_DIR+args.model+'_KAMINSKYI_plt_test_LSTM_'+str(args.time_gap)+'s_'+str(args.num_images)+':10_im_to_pred_pitch_final'+'.png')
# plt.savefig('pitch_test.png')
plt.show()
