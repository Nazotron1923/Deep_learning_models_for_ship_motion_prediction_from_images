"""
transform the data obtained from blender.py to the one which can be used for training
"""
import json
import argparse
import sys
import os
import numpy as np

dirpath = os.getcwd()
sys.path.append(dirpath)
FRAME_STEP = 1
INPUT_FILE_NAME = '/paras_origin.json'
OUTPUT_FILE_NAME = '/labels_'
def transformDate(input_folder, output_folder, time_gaps, episode_number, frame_interval):
	print("OS pwd - ", os.getcwd())
	max_min_dict = { 	'min_pitch' : np.inf,
						'min_roll' : np.inf,
						'max_pitch' : -np.inf,
						'max_roll' : -np.inf
					}

	for episode in range(1, episode_number+1):
		inputFile = input_folder + str(episode) + INPUT_FILE_NAME
		labels = json.load(open(inputFile))
		# string list
		TOTAL_FRAME = list(labels.keys())
		# for python2 string->int
		# TOTAL_FRAME = map(int, TOTAL_FRAME)
		# for python3 string->int
		TOTAL_FRAME = list(map(int, TOTAL_FRAME))
		MAX_FRAME = max(TOTAL_FRAME)
		MIN_FRAME = min(TOTAL_FRAME)
		print("transforming data ...")
		# add time gap to data

		# create all time gaps
		for tg in time_gaps:
			ADD_STEP = int ((24/frame_interval)*tg)
			print("episode ---______________________ ", episode)
			# for i in range(len(labels)):
			for i in range(MIN_FRAME,MAX_FRAME+1):
				if labels.get(str(i)) != None and i < (MAX_FRAME - ADD_STEP):
					labels[str(i)] = labels[str(i+ADD_STEP)]
			# delete frames which exceed total frame
			for i in range(MAX_FRAME - ADD_STEP, MAX_FRAME+1):
				if labels.get(str(i)) != None:
					labels.pop(str(i))
			# use only pitch and roll to train, delete yaw
			for key, value in labels.items():
				labels[key] = value[:-1]
				if labels[key][1] < max_min_dict['min_pitch']:
					max_min_dict['min_pitch'] = labels[key][1]
					if(labels[key][1] < - 90.0):
						print('position--------------------------------------------------------- ', key)
				if labels[key][1] > max_min_dict['max_pitch']:
					max_min_dict['max_pitch'] = labels[key][1]
					if(labels[key][1] > 90.0):
						print('position--------------------------------------------------------- ', key)
				if labels[key][0] < max_min_dict['min_roll']:
					max_min_dict['min_roll'] = labels[key][0]
					if(labels[key][0] < - 90.0):
						print('position--------------------------------------------------------- ', key)
				if labels[key][0] > max_min_dict['max_roll']:
					max_min_dict['max_roll'] = labels[key][0]
					if(labels[key][0] >  90.0):
						print('position--------------------------------------------------------- ', key)

			output = output_folder + str(episode) + OUTPUT_FILE_NAME + str(tg) + '.json'
			json.dump(labels,open(output,'w'))
			print("transform data step - ", tg)
			labels = json.load(open(inputFile))




	print("\ntransform data done !")
	print(max_min_dict)
	json.dump(max_min_dict,open('min_max_statistic_320_episodes.json' ,'w'))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform parameters')
    parser.add_argument('-i', '--InputFolder', help='Input folders names prefix', default="./test_4_episode_", type=str)
    parser.add_argument('-o', '--OutputFolder', help='Output folder name prefix', default='./test_4_episode_', type=str)
    parser.add_argument('-e', '--EpisodeNumber', help='Number of episodes in generation', default= 100, type=int)  			# to know how many episodes are created
    parser.add_argument('-t', '--TimeGaps', help='List of Time gaps', default=[0], type=list)  				# to test more gaps
    parser.add_argument('-f', '--FrameIntervalParameter', help='Specify the FRAME_INTERVAL when rendering', default=12, type=int)	# FRAME_INTERVAL
    args = parser.parse_args()

    transformDate(input_folder=args.InputFolder, output_folder=args.OutputFolder, time_gaps=args.TimeGaps, episode_number = args.EpisodeNumber, frame_interval = args.FrameIntervalParameter)
