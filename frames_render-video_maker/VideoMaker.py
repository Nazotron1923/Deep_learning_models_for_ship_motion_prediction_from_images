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

img_array = []
act_path = os.getcwd()

def VideoMaker(desktop_path):
    print(act_path)
    for folder in next(os.walk(desktop_path))[1]:
        
        folder_path = desktop_path +'/'+ folder
        
        if os.path.isdir(folder_path + '/data'):
            for episode in next(os.walk(folder_path + '/data'))[1]:
                #os.chdir(folder_path + '/data/' + episode)
                sys.path.append(folder_path + '/data/' + episode)
                print(os.getcwd())
                if os.path.isfile('parameters.py'):
                    from parameters import FRAME_INTERVAL
                    frames = glob.glob(folder_path + '/data/' + episode + '/*.png')
                    
                    for filename in frames:
                        img = cv2.imread(filename)
                        height, width, layers = img.shape
                        size = (width,height)
                        img_array.append(img)
                    out = cv2.VideoWriter(episode+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25/FRAME_INTERVAL, size)
               
                    os.chdir(act_path)
                    for i in range(len(img_array)):
                        out.write(img_array[i])
                    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify desktop path')
    parser.add_argument('-d', '--desktop', help='Desktop path', type=str)
    args = parser.parse_args()

    VideoMaker(desktop_path=args.desktop)
