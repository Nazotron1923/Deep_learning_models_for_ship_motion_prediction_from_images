"""
File containing all the constants used in the different files
"""
# Main Constants
CAMERA_RESOLUTION = (96, 54)
# Regions of interest
MAX_WIDTH = CAMERA_RESOLUTION[0]
MAX_HEIGHT = CAMERA_RESOLUTION[1]
ROI = [0, 0, MAX_WIDTH, MAX_HEIGHT]

# dataset sample frame interval
FRAME_INTERVAL = 12

# Training
FACTOR = 1  # Resize factor
INPUT_HEIGHT = ROI[3] // FACTOR
INPUT_WIDTH = ROI[2] // FACTOR

# folder with ALL results
RES_DIR = "./Pre/results/"

# parametes of sequence
LEN_SEQ = 60
SEQ_PER_EPISODE_C = 6

# Random seed
RANDS = 13
