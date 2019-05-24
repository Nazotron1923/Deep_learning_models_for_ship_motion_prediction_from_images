import math
#size of the simulation
frame_end = 20000
#parameters of the path to follow
cir_x = 400
cir_y = 400
#Parameters of the camera
heigth = 5
rot_x = 76 * math.pi / 180 # <90 go camera down
rot_y = 0 * math.pi / 180
rot_z = 180 * math.pi / 180	
#parameters of the sea
size = 0.1
spatial_size = 10000
depth = 500
resolution = 14
choppiness = [0.6, 0.7, 0.8]
wave_scale = [4, 5, 6]
wave_scale_min = 0.01
wind_velocity = [30, 40, 50]
wave_alignment = 5
random_seed = [1, 2, 3, 4]
#for the render
RENDER = True
samples_render = 135 #135
FRAME_INTERVAL = 12  
START_FRAME = 0
END_FRAME = 4800
resolution_x = 96 
resolution_y = 54

#for file make prosess
version = "test_1_"
