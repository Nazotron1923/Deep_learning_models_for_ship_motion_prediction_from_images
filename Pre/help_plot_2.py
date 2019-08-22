"""
this code is used to plot evolution MSE for pitch and roll 
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np



CNN_LSTM_image_encoder_PR_encoder_decoder_MSE_roll_in_time = np.array([0.00035159, 0.000329, 0.00035358, 0.00034741, 0.00034888, 0.00038025,
 0.0003636, 0.00034148, 0.00037102, 0.00036764, 0.00034107, 0.00033349,
 0.00034112, 0.00035879, 0.00033773, 0.0003394,  0.00032551, 0.0003472,
 0.00033376, 0.00035027, 0.00034711, 0.00037118, 0.0004029, 0.00042698])
CNN_LSTM_image_encoder_PR_encoder_decoder_MSE_pitch_in_time = np.array([0.00040733, 0.00039321, 0.00039035, 0.00040324, 0.00040132, 0.00042157,
 0.00043905, 0.00042345, 0.00042061, 0.00041076, 0.0004268,  0.00041901,
 0.00040015, 0.00040097, 0.0003975,  0.00038725, 0.00040378, 0.00041345,
 0.00045092, 0.00043964, 0.00045015, 0.00044738, 0.00046267, 0.00047911]
)
CNN_LSTM_image_encoder_PR_encoder_decoder_MSE_avg = np.array([0.00075892, 0.00072221, 0.00074392, 0.00075066, 0.0007502,  0.00080183,
 0.00080265, 0.00076494, 0.00079163, 0.0007784,  0.00076786, 0.0007525,
 0.00074127, 0.00075976, 0.00073523, 0.00072665, 0.0007293,  0.00076064,
 0.00078468, 0.00078991, 0.00079726, 0.00081856, 0.00086557, 0.00090609]
)
CNN_LSTM_image_encoder_PR_encoder_decoder_MSE_test_res = np.array([0.00077503, 0.00077503, 0.00077503, 0.00077503, 0.00077503, 0.00077503,
 0.00077503, 0.00077503, 0.00077503, 0.00077503, 0.00077503, 0.00077503,
 0.00077503, 0.00077503, 0.00077503, 0.00077503, 0.00077503, 0.00077503,
 0.00077503, 0.00077503, 0.00077503, 0.00077503, 0.00077503, 0.00077503]
)

CNN_LSTM_encoder_decoder_images_MSE_roll_in_time = np.array([0.0002824,  0.00028308, 0.00028434, 0.00028929, 0.00029328, 0.00031329,
 0.00032473, 0.00031851, 0.00030959, 0.00031666, 0.00029769, 0.00032508,
 0.00033326, 0.000346  , 0.00030752, 0.00029677, 0.0002928 , 0.00028776,
 0.00030947, 0.00028676, 0.00029316, 0.00030073, 0.00030509, 0.00033229]
)
CNN_LSTM_encoder_decoder_images_MSE_pitch_in_time = np.array([0.00035714, 0.00033579, 0.00035508, 0.00036832, 0.0003714,  0.00036137,
 0.00037377, 0.0003626 , 0.00035423, 0.00037483, 0.00035598, 0.00035023,
 0.00035161, 0.00035478, 0.00038192, 0.0003888,  0.00041147, 0.0004166,
 0.00040492, 0.00042253, 0.00041407, 0.00041423, 0.0004379,  0.00045569]
)
CNN_LSTM_encoder_decoder_images_MSE_avg = np.array([0.00063953, 0.00061887, 0.00063942, 0.00065761, 0.00066468, 0.00067466,
 0.0006985,  0.00068111, 0.00066382, 0.0006915,  0.00065367, 0.0006753,
 0.00068486, 0.00070078, 0.00068944, 0.00068557, 0.00070426, 0.00070436,
 0.0007144 , 0.00070929, 0.00070722, 0.00071495, 0.00074299, 0.00078798]
)
CNN_LSTM_encoder_decoder_images_MSE_test_res = np.array([0.0006877, 0.0006877, 0.0006877, 0.0006877, 0.0006877, 0.0006877, 0.0006877,
 0.0006877, 0.0006877, 0.0006877, 0.0006877, 0.0006877, 0.0006877, 0.0006877,
 0.0006877, 0.0006877, 0.0006877, 0.0006877, 0.0006877, 0.0006877, 0.0006877,
 0.0006877, 0.0006877, 0.0006877]
)

CNN_LSTM_encoder_decoder_images_PR_MSE_roll_in_time = np.array([0.000286,   0.0002731,  0.00026912, 0.00029316, 0.00029453, 0.00031095,
 0.00029481, 0.00029939, 0.00032485, 0.0003151 , 0.00032268, 0.0003233,
 0.00032344, 0.0003327 , 0.00033053, 0.00032138, 0.00032899, 0.00035189,
 0.00033645, 0.00033449, 0.00035655, 0.00035251, 0.00036215, 0.00037532]
)
CNN_LSTM_encoder_decoder_images_PR_MSE_pitch_in_time = np.array([0.00039015, 0.00037639, 0.00035298, 0.00035839, 0.00037423, 0.00036937,
 0.00037296, 0.00035983, 0.00035928, 0.00036656, 0.00035602, 0.00034089,
 0.00034644, 0.0003778 , 0.00037426, 0.00037673, 0.00037947, 0.00036773,
 0.00035423, 0.00035743, 0.00037102, 0.00035878, 0.00037209, 0.00039235]
)
CNN_LSTM_encoder_decoder_images_PR_MSE_avg = np.array( [0.00067615, 0.00064949, 0.00062211, 0.00065155, 0.00066876, 0.00068032,
 0.00066777, 0.00065922, 0.00068414, 0.00068166, 0.00067869, 0.00066418,
 0.00066987, 0.0007105 , 0.00070479, 0.0006981 , 0.00070846, 0.00071963,
 0.00069068, 0.00069192, 0.00072756, 0.00071129, 0.00073424, 0.00076767]
)
CNN_LSTM_encoder_decoder_images_PR_MSE_test_res = np.array([0.00068828, 0.00068828, 0.00068828, 0.00068828, 0.00068828, 0.00068828,
 0.00068828, 0.00068828, 0.00068828, 0.00068828, 0.00068828, 0.00068828,
 0.00068828, 0.00068828, 0.00068828, 0.00068828, 0.00068828, 0.00068828,
 0.00068828, 0.00068828, 0.00068828, 0.00068828, 0.00068828, 0.00068828]
)

LSTM_encoder_decoder_PR_MSE_roll_in_time = np.array([0.00454642, 0.00459153, 0.00461934, 0.00474111, 0.00490542, 0.00488522,
 0.00489724, 0.00498072, 0.00511352, 0.00526438, 0.00531938, 0.00524217,
 0.00525478, 0.00528606, 0.00533018, 0.00520502, 0.00536538, 0.00526944,
 0.00528865, 0.0051717,  0.00512693, 0.00516668, 0.0051589,  0.00517767])
LSTM_encoder_decoder_PR_MSE_pitch_in_time = np.array([0.0053205 , 0.00576082, 0.00582789, 0.00610407, 0.00627336, 0.00657175,
 0.00638995, 0.00649575, 0.00664366, 0.006707,   0.00663234, 0.00677326,
 0.00689362, 0.00688079, 0.00676608, 0.00677617, 0.00692191, 0.00698005,
 0.0070963,  0.00711009, 0.0070644,  0.00709673, 0.00713486, 0.00702159]
)
LSTM_encoder_decoder_PR_MSE_avg = np.array([0.00986692, 0.01035235, 0.01044723, 0.01084519, 0.01117877, 0.01145697,
 0.01128719, 0.01147647, 0.01175718, 0.01197139, 0.01195172, 0.01201544,
 0.0121484 , 0.01216686, 0.01209626, 0.01198119, 0.01228729, 0.01224949,
 0.01238494, 0.01228179, 0.01219133, 0.01226342, 0.01229376, 0.01219926]
)
LSTM_encoder_decoder_PR_MSE_test_res = np.array([0.01171462, 0.01171462, 0.01171462, 0.01171462, 0.01171462, 0.01171462,
 0.01171462, 0.01171462, 0.01171462, 0.01171462, 0.01171462, 0.01171462,
 0.01171462, 0.01171462, 0.01171462, 0.01171462, 0.01171462, 0.01171462,
 0.01171462, 0.01171462, 0.01171462, 0.01171462, 0.01171462, 0.01171462]
)

x = np.linspace(1,24, 24, dtype =int)

font = {'size'   : 20}

plt.figure(1)
# resize pic to show details
plt.figure(figsize=(24, 16))
# plt.plot(x, LSTM_encoder_decoder_PR_MSE_roll_in_time, 'r-', label='LSTM encoder decoder PR model - MSE roll evolution [BASELINE]')
# plt.plot(x, LSTM_encoder_decoder_PR_MSE_pitch_in_time, 'r--', label='LSTM encoder decoder PR model - MSE pitch evolution [BASELINE]')
plt.plot(x, CNN_LSTM_encoder_decoder_images_PR_MSE_roll_in_time, 'r-', label='CNN LSTM encoder decoder images PR model - MSE roll',  linewidth=3)
plt.plot(x, CNN_LSTM_encoder_decoder_images_PR_MSE_pitch_in_time, 'r--', label='CNN LSTM encoder decoder images PR model - MSE pitch',  linewidth=3)
plt.plot(x, CNN_LSTM_image_encoder_PR_encoder_decoder_MSE_roll_in_time, 'g-', label='CNN LSTM image encoder PR encoder decoder model - MSE roll',  linewidth=3)
plt.plot(x, CNN_LSTM_image_encoder_PR_encoder_decoder_MSE_pitch_in_time, 'g--', label='CNN LSTM image encoder PR encoder decoder model - MSE pitch',  linewidth=3)
plt.plot(x, CNN_LSTM_encoder_decoder_images_MSE_roll_in_time, 'b-', label='CNN LSTM encoder decoder images model - MSE roll',  linewidth=3)
plt.plot(x, CNN_LSTM_encoder_decoder_images_MSE_pitch_in_time, 'b--', label='CNN LSTM encoder decoder images model - MSE pitch',  linewidth=3)

plt.title("Test Data\nMSE - predicted frames [2 fps]", fontsize=24)
plt.xlabel("Predicted frames", fontsize=24)
plt.ylabel("MSE - loss function", fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid(True)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06),
          fancybox=True, shadow=True, ncol=3, prop={'size': 16})
str_t = str('MSE_evolution_PR_3_models.png')
plt.savefig(str_t)
plt.show()

plt.figure(2)
# resize pic to show details
plt.figure(figsize=(24, 16))
# plt.plot(x, LSTM_encoder_decoder_PR_MSE_avg, 'r-', label='LSTM encoder decoder PR model - avarage MSE pitch and roll [BASELINE]')
# plt.plot(x, LSTM_encoder_decoder_PR_MSE_test_res, 'r--', label='LSTM encoder decoder PR model - result [BASELINE]')
plt.plot(x, CNN_LSTM_encoder_decoder_images_PR_MSE_avg, 'r-', label='CNN LSTM encoder decoder images PR model - (MSE pitch + MSE roll)',  linewidth=3)
plt.plot(x, CNN_LSTM_encoder_decoder_images_PR_MSE_test_res, 'r--', label='CNN LSTM encoder decoder images PR model - average over sequence (MSE pitch + MSE roll)',  linewidth=3)
plt.plot(x, CNN_LSTM_image_encoder_PR_encoder_decoder_MSE_avg, 'g-', label='CNN LSTM image encoder PR encoder decoder model - (MSE pitch + MSE roll)',  linewidth=3)
plt.plot(x, CNN_LSTM_image_encoder_PR_encoder_decoder_MSE_test_res, 'g--', label='CNN LSTM image encoder PR encoder decoder model - average over sequence (MSE pitch + MSE roll)',  linewidth=3)
plt.plot(x, CNN_LSTM_encoder_decoder_images_MSE_avg, 'b-', label='CNN LSTM encoder decoder images model - (MSE pitch + MSE roll)',  linewidth=3)
plt.plot(x, CNN_LSTM_encoder_decoder_images_MSE_test_res, 'b--', label='CNN LSTM encoder decoder images model - average over sequence (MSE pitch + MSE roll)',  linewidth=3)



plt.title("Test data\nMSE - predicted frames [2 fps]", fontsize=24)
plt.xlabel("Predicted frames", fontsize=24)
plt.ylabel("MSE - loss function", fontsize=24)
plt.grid(True)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(loc='upper left', prop={'size': 22})
str_t = str('avg_MSE_evolution_PR_3_models.png')
plt.savefig(str_t)
plt.show()

# tmp_im = "train_{}_using_{}_s_to_predict_{}_s_lr_{}_{}".format(args.model_type, args.use_sec, args.time_gap, args.learning_rate, args.date)
# str_t = str(args.prediction_file+tmp_im+'/MSE_evolution_denorm.png')
# plt.savefig(str_t)
