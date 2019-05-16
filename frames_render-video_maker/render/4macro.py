#/blender/blender --background --python 4macro.py
import bpy
from bpy.app.handlers import persistent
import bpy, bgl, blf,sys
from bpy import data, ops, props, types, context
from math import degrees
import json
import sys
import os
import shutil
from datetime import datetime
import time
import pickle

dirpath = os.getcwd()
sys.path.append(dirpath)

from parameters import *



#this macro needs the file ocean_render.blend


bpy.ops.wm.open_mainfile(filepath='./ocean_render_2.blend')

@persistent
def load_handler(dummy):
	print("Load Handler:", bpy.data.filepath)

	bpy.data.scenes["Scene"].cycles.samples = samples_render
	
	ocean = bpy.data.objects['Ocean']
	
	bpy.context.scene.frame_end = frame_end
	#clear the keyframes
	ocean.animation_data_clear()
	#set the keyframes in order to mqke the movement of the sea
	ocean.modifiers["Ocean"].time = 1
	ocean.keyframe_insert(data_path='modifiers["Ocean"].time',frame=1.0)
	ocean.modifiers["Ocean"].time = 10
	ocean.keyframe_insert(data_path='modifiers["Ocean"].time',frame=250.0)
	bpy.data.objects['Ocean']
	#modify the parameters of the sea 					recomended parameters
	ocean.modifiers["Ocean"].size = size 				#1
	ocean.modifiers["Ocean"].spatial_size = spatial_size		#50
	ocean.modifiers["Ocean"].depth = depth				#200
	ocean.modifiers["Ocean"].resolution = resolution 			#15
	#ocean.modifiers["Ocean"].choppiness = choppiness			#0.7
	#ocean.modifiers["Ocean"].wave_scale = wave_scale				#2
	ocean.modifiers["Ocean"].wave_scale_min = wave_scale_min		#0.01
	#ocean.modifiers["Ocean"].wind_velocity = wind_velocity			#30
	ocean.modifiers["Ocean"].frame_start = 1			#1
	ocean.modifiers["Ocean"].frame_end = 250			#250
	ocean.modifiers["Ocean"].wave_alignment = wave_alignment	   #min 0 max 10
	#ocean.modifiers["Ocean"].random_seed = random_seed	   #min 0 max 10

	#bpy.context.object.modifiers["Ocean"].use_foam = False

	#set the movement of the sea as linear
	ocean.animation_data.action.fcurves[0].extrapolation='LINEAR'

	bpy.ops.object.select_all(action='DESELECT')

	#create the path to follow (the circle)
	bpy.ops.curve.primitive_bezier_circle_add(radius=1, view_align=False, enter_editmode=True, location=(0,0,0), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
	bpy.context.scene.objects.active = bpy.data.objects['BezierCircle'] 

	bpy.ops.object.editmode_toggle()
	bpy.ops.transform.resize(value=(cir_x, cir_y, 1), constraint_axis=(False, False, False), constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED', proportional_edit_falloff='SMOOTH', proportional_size=1)
	#bpy.data.objects.["Plane"].constraints["Follow Path"].use_curve_follow = True


	#set the parameters of the path 
	bpy.data.curves["BezierCircle"].eval_time = 1
	bpy.data.curves["BezierCircle"].keyframe_insert(data_path='eval_time', frame = 1)
	bpy.data.curves["BezierCircle"].eval_time = 100
	bpy.data.curves["BezierCircle"].keyframe_insert(data_path='eval_time', frame = 10000)
	bpy.data.curves["BezierCircle"].animation_data.action.fcurves[0].extrapolation='LINEAR'


	#create the plane
	bpy.ops.mesh.primitive_plane_add(radius=1, view_align=False, enter_editmode=False, location=(0,0,0), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
	bpy.context.object.scale[0] = 1
	bpy.context.object.scale[1] = 1
	bpy.context.object.scale[2] = 1
	bpy.ops.material.new()

	#divide the plane
	bpy.ops.object.editmode_toggle()
	bpy.ops.mesh.subdivide(smoothness=0)
	bpy.ops.mesh.subdivide(smoothness=0)
	bpy.ops.mesh.subdivide(smoothness=0)
	bpy.ops.mesh.subdivide(smoothness=0)


	bpy.ops.object.vertex_group_add()
	bpy.ops.object.vertex_group_assign()
	bpy.ops.object.editmode_toggle()


	#make the plane follow the circle
	bpy.ops.object.constraint_add(type='FOLLOW_PATH')
	bpy.context.object.constraints["Follow Path"].target = bpy.data.objects["BezierCircle"]
	bpy.context.object.constraints["Follow Path"].use_curve_follow = True
	bpy.ops.object.modifier_add(type='SHRINKWRAP')
	bpy.context.object.modifiers["Shrinkwrap"].target = bpy.data.objects["Ocean"]


	#create a cube to follow the path and to glue the camera
	bpy.ops.mesh.primitive_cube_add(radius=1, view_align=False, enter_editmode=False, location=(0, 0, 0), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
	bpy.context.object.scale[0] = 0.3
	bpy.context.object.scale[1] = 0.3
	bpy.context.object.scale[2] = 0.3
	bpy.ops.object.select_all(action='DESELECT')


	#create the camera and glue it to the cube
	

	bpy.ops.object.camera_add(view_align=True, enter_editmode=False, location=(0, 0, 0), rotation=(0,0,0), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))
	bpy.context.scene.objects.active = bpy.data.objects['Camera'] 

	bpy.context.object.rotation_euler[0] = rot_x
	bpy.context.object.rotation_euler[1] = rot_y
	bpy.context.object.rotation_euler[2] = rot_z

	bpy.ops.object.constraint_add(type='CHILD_OF')
	bpy.context.object.constraints["Child Of"].target = bpy.data.objects["Cube"]
	bpy.ops.transform.translate(value=(0, 0, heigth), constraint_axis=(False, False, True), constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED', proportional_edit_falloff='SMOOTH', proportional_size=1, release_confirm=True)
	bpy.context.object.constraints["Child Of"].use_rotation_y = False
	bpy.context.object.constraints["Child Of"].use_rotation_x = False


	#finally glue the cube and the plane
	bpy.context.scene.objects.active = bpy.data.objects['Cube'] 
	bpy.ops.object.constraint_add(type='COPY_LOCATION')
	bpy.ops.object.constraint_add(type='COPY_ROTATION')
	bpy.context.object.constraints["Copy Location"].target = bpy.data.objects["Plane"]
	bpy.context.object.constraints["Copy Rotation"].target = bpy.data.objects["Plane"]
	bpy.context.object.constraints["Copy Location"].subtarget = "Group"
	bpy.context.object.constraints["Copy Rotation"].subtarget = "Group"

	#render propierties
	bpy.context.scene.render.engine = 'CYCLES'
	scene = context.scene
	scene.render.resolution_x = resolution_x * 2
	scene.render.resolution_y = resolution_y * 2

bpy.app.handlers.load_post.append(load_handler)
bpy.ops.wm.open_mainfile(filepath='./ocean_render_2.blend')
ocean = bpy.data.objects['Ocean']

l0 = 0
fold = 1
for random_seed1 in random_seed:
	for wind_velocity1 in wind_velocity:
		for wave_scale1 in wave_scale:
			for choppiness1 in choppiness:
				ocean.modifiers["Ocean"].wave_scale = wave_scale1
				ocean.modifiers["Ocean"].wind_velocity = wind_velocity1			
				ocean.modifiers["Ocean"].random_seed = random_seed1	   
				ocean.modifiers["Ocean"].choppiness = choppiness1

				if RENDER == True:
					#folder creation and managing


					today = datetime.now()

					os.mkdir("./data/" + today.strftime('%Y%m%d%H%M%S'))

					BOAT = 'Cube'
					CAMERA = 'Camera'
					IMG_DIR="./data/" + today.strftime('%Y%m%d%H%M%S')

					print('\nPrint Scenes...')
					shutil.copy('parameters.py',IMG_DIR)
					paras = {}
					context = bpy.context
					scene = context.scene
					currentCameraObj = bpy.data.objects[CAMERA]

					scene.camera = currentCameraObj
					param = {'wave_scale': wave_scale1, 'seed': random_seed1, 'wind_velocity': wind_velocity1, 'choppiness': choppiness1}
					jsPar = json.dumps(param)
					filePar = open(IMG_DIR+"/"+'parameters.json', 'w')
					filePar.write(jsPar)
					filePar.close()

					l1 = len(random_seed)
					l2 = len(wind_velocity)
					l3 = len(wave_scale)
					l4 = len(choppiness)
					
					for i in range(START_FRAME,int(END_FRAME/FRAME_INTERVAL)+1):
						Time_st = time.time()
						print('current frame:', i)
						# set current frame
						scene.frame_set(i*FRAME_INTERVAL)
						scene.render.filepath = IMG_DIR + "/" + str(i*FRAME_INTERVAL)
						bpy.ops.render.render(write_still=True)
						# get paras (degree)
						loc, rot, scale = bpy.data.objects[BOAT].matrix_world.decompose()
						rot = rot.to_euler()
						rot = list(degrees(a) for a in rot)
						paras.update({i*FRAME_INTERVAL:rot})
						jsObj = json.dumps(paras)
						fileObject = open(IMG_DIR+"/"+'paras_origin.json', 'w')
						fileObject.write(jsObj)
						fileObject.close()
						Time_end = time.time()

						Time_dif = Time_end - Time_st

						l0+=1
						print(l0)	
						Time_rem = Time_dif * (l1 * l2 * l3 * END_FRAME/FRAME_INTERVAL - l0)
						print('##############################################################################################')
						print('##############################################################################################')

						print('The time remaining is approximately [hours]:', Time_rem/3600) 
						print('This is the folder:' , fold ,' from ', l1 * l2 * l3*l4)

						print('##############################################################################################')
						print('##############################################################################################')
					fold+=1
					choppiness_i=[i for i,x in enumerate(choppiness) if x==choppiness1]
					wave_scale_i=[i for i,x in enumerate(wave_scale) if x==wave_scale1]
					wind_velocity_i=[i for i,x in enumerate(wind_velocity) if x==wind_velocity1]
					random_seed_i=[i for i,x in enumerate(random_seed) if x==random_seed1]

					choppiness_save=choppiness[choppiness_i[0]:]
					wave_scale_save=wave_scale[wave_scale_i[0]:]
					wind_velocity_save=wind_velocity[wind_velocity_i[0]:]
					random_seed_save=random_seed[random_seed_i[0]:]
					with open('log.pkl', 'wb') as f:
						pickle.dump([choppiness_save, wave_scale_save, wind_velocity_save, random_seed_save], f)

	print(paras)
	print('Done!')
	
