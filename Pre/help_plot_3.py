"""
this code is used to plot # CNN_stack_FC model and # CNN_stack_PR_FC model tests figures
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# CNN_stack_PR_FC model                                                    5s                        10s                      15s
train_CNN_stack_PR_FC_using_10_s_lr_5e_05 = np.array([0.005790986916205536, 0.008295571450920155 ,0.008771926863119006])
train_CNN_stack_PR_FC_using_10_s_lr_0_0005 = np.array([0.011257842706982046, 0.009126499140014251,0.009524343428590024])
train_CNN_stack_PR_FC_using_10_s_lr_0_0001 = np.array([0.0043358881763803464, 0.006081611723251019,0.006629019005534549])

train_CNN_stack_PR_FC_using_8_s_lr_5e_05 = np.array([0.005093533463438908, 0.006939627487083961, 0.007963413533208699])
train_CNN_stack_PR_FC_using_8_s_lr_0_0005 = np.array([0.007063912561942084, 0.006963715579457067, 0.007309839091311884])
train_CNN_stack_PR_FC_using_8_s_lr_0_0001 = np.array([0.003799249346825647, 0.004833479584515865, 0.0058962320146183])

train_CNN_stack_PR_FC_using_5_s_lr_5e_05 = np.array([0.005144534817713491, 0.006784352724083719, 0.007777513559360275])
train_CNN_stack_PR_FC_using_5_s_lr_0_0005 = np.array([0.006804681293284918, 0.009356206365950168, 0.008211554467437808])
train_CNN_stack_PR_FC_using_5_s_lr_0_0001 = np.array([0.0040882718640979027, 0.00541933258670937, 0.006436676437783233])

# CNN_stack_FC model
# train_CNN_stack_FC_using_10_s_lr_5e_05 = np.array([0.005312037396621197, 0.006933201490802334 ,0.0076734735828606375])
# train_CNN_stack_FC_using_10_s_lr_0_0005 = np.array([0.007378957957584173, 0.008725171165659707, 0.00861803922762579])
# train_CNN_stack_FC_using_10_s_lr_0_0001 = np.array([0.00397565506616647, 0.005426044313316333, 0.0065448140834477985])
#
# train_CNN_stack_FC_using_8_s_lr_5e_05 = np.array([0.005269838621627474, 0.006478569188498352, 0.007581346979765058])
# train_CNN_stack_FC_using_8_s_lr_0_0005 = np.array([0.006799398988828455, 0.0077597480634913635, 0.008804853793168277])
# train_CNN_stack_FC_using_8_s_lr_0_0001 = np.array([0.0041365403663612115, 0.005252795767757976, 0.006674040893190785])
#
# train_CNN_stack_FC_using_5_s_lr_5e_05 = np.array([0.005260791404314501, 0.007000325543745218, 0.007687830097932854])
# train_CNN_stack_FC_using_5_s_lr_0_0005 = np.array([0.007153622360657091, 0.009097261906034597, 0.009357064926956335])
# train_CNN_stack_FC_using_5_s_lr_0_0001 = np.array([0.004794604117151997, 0.006078774554897909, 0.006917447633737617])

x = np.array([10, 20, 30], dtype =int)
font = {'size'   : 20}

plt.figure(1)
# resize pic to show details
plt.figure(figsize=(8, 5))
plt.plot(x, train_CNN_stack_PR_FC_using_10_s_lr_5e_05, 'r-', label='Past window size - 10 sec')
plt.plot(x, train_CNN_stack_PR_FC_using_8_s_lr_5e_05, 'g-', label='Past window size - 8 sec')
plt.plot(x, train_CNN_stack_PR_FC_using_5_s_lr_5e_05, 'b-', label='Past window size - 5 sec')

plt.title("MSE test - predicted frames [2 fps]\n Lr = 0.00005", fontsize=18)
plt.xlabel("Predicted frames", fontsize=16)
plt.ylabel("MSE - loss function", fontsize=16)
plt.grid(True)
plt.legend(loc='lower right', prop={'size': 13})
str_t = str('MSE_evolution_CNN_stack_PR_FC_lr_5e_05.png')
plt.savefig(str_t)
plt.show()

plt.figure(2)
# resize pic to show details
plt.figure(figsize=(8, 5))
plt.plot(x, train_CNN_stack_PR_FC_using_10_s_lr_0_0005, 'r-', label='Past window size - 10 sec')
plt.plot(x, train_CNN_stack_PR_FC_using_8_s_lr_0_0005, 'g-', label='Past window size - 8 sec')
plt.plot(x, train_CNN_stack_PR_FC_using_5_s_lr_0_0005, 'b-', label='Past window size - 5 sec')

plt.title("MSE test - predicted frames [2 fps]\n Lr = 0.0005", fontsize=18)
plt.xlabel("Predicted frames", fontsize=16)
plt.ylabel("MSE - loss function", fontsize=16)
plt.grid(True)
plt.legend(loc='lower right', prop={'size': 13})
str_t = str('MSE_evolution_CNN_stack_PR_FC_lr_5e_04.png')
plt.savefig(str_t)
plt.show()


plt.figure(3)
# resize pic to show details
plt.figure(figsize=(8, 5))
plt.plot(x, train_CNN_stack_PR_FC_using_10_s_lr_0_0001, 'r-', label='Past window size - 10 sec')
plt.plot(x, train_CNN_stack_PR_FC_using_8_s_lr_0_0001, 'g-', label='Past window size - 8 sec')
plt.plot(x, train_CNN_stack_PR_FC_using_5_s_lr_0_0001, 'b-', label='Past window size - 5 sec')

plt.title("MSE test - predicted frames [2 fps]\n Lr = 0.0001", fontsize=18)
plt.xlabel("Predicted frames", fontsize=16)
plt.ylabel("MSE - loss function", fontsize=16)
plt.grid(True)
plt.legend(loc='lower right', prop={'size': 13})
str_t = str('MSE_evolution_CNN_stack_PR_FC_lr_1e_04.png')
plt.savefig(str_t)
plt.show()
