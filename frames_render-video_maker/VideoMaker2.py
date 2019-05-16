 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:23:49 2019

@author: manuelbatet
"""

import os
import argparse
import cv2
import glob
import sys

#save actual path
act_path = os.getcwd()


def VideoMaker(desktop_path):
    #check all the folders in the directory
    for folder in next(os.walk(desktop_path))[1]:
        #path of each folder
        folder_path = desktop_path +'/'+ folder
        
        #if there is any folder called data inside the folder_path continue
        if os.path.isdir(folder_path + '/data'):
            #inside the folder_path there should be the episodes' folders
            for episode in next(os.walk(folder_path + '/data'))[1]:
                #change the chdir in order to check and import parameters.py
                os.chdir(folder_path + '/data/' + episode +'/')
                print('checking folder: ' + folder_path + '/data/' + episode)
                sys.path.append(folder_path + '/data/' + episode)
                #if parameters exist continue
                if os.path.isfile('parameters.py'):
                    #import the frame interval from parameters
                    from parameters import FRAME_INTERVAL
                    #get a list of the frames
                    frames =sorted( glob.glob(folder_path + '/data/' + episode + '/*.png') )
                    #initialize the img_array
                    img_array=[]
                    for filename in frames:
                        
                        print('checking frame:' + filename)
                        img = cv2.imread(filename)
                        height, width, layers = img.shape
                        size = (width,height)
                        img_array.append(img)
                    #if there were some frames continue    
                    if len(frames)>0: 
                        os.chdir(act_path)
                        out = cv2.VideoWriter(episode+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 20/FRAME_INTERVAL, size)
                        for i in range(len(img_array)):
                            out.write(img_array[i])
                        out.release()
                        print('videos have been saved in ' + act_path)
                        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify desktop path')
    parser.add_argument('-d', '--desktop', help='Desktop path', type=str)
    args = parser.parse_args()

    VideoMaker(desktop_path=args.desktop)
