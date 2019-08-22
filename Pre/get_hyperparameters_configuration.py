import numpy as np
from math import log, sqrt
from time import time
from pprint import pprint

from hyperopt import hp
from hyperopt.pyll.stochastic import sample

iters_per_iteration = 1

# handle floats which should be integers
# works with flat params
def handle_integers( params ):

	new_params = {}
	for k, v in params.items():
		if type( v ) == float and int( v ) == v:
			new_params[k] = int( v )
		else:
			new_params[k] = v

	return new_params



space = {
	# 'opt': hp.choice('opt', ['adam', 'sgd']),
	'encoder_latent_vector': hp.quniform( 'encoder_latent_vector', 500, 1500, 1 ),
	'decoder_latent_vector': hp.quniform( 'decoder_latent_vector', 250, 500, 1 ),
	'learning_rate': hp.uniform( 'learning_rate', 9e-6, 2e-3),
	'weight_decay': hp.uniform( 'weight_decay', 9e-5, 3e-3),
	# 'use_sec': hp.quniform( 'use_sec', 7, 10, 1 ),

}

def get_params(args):

	params = sample( space )
	params = handle_integers( params )

	params['train_folder'] = args.train_folder
	params['batchsize'] = args.batchsize
	params['opt'] = args.opt
	params['seed'] = args.seed
	params['cuda'] = args.cuda
	params['load_weight'] = args.load_model
	params['load_weight_date'] = args.load_weight_date
	params['model_type'] = args.model_type
	params['time_to_predict'] = args.time_to_predict
	params['test_dir'] = args.test_dir
	params['frame_interval'] = args.frame_interval
	params["use_n_episodes"] = args.use_n_episodes
	params['use_sec'] = args.use_sec

	return params


space_VAE = {
	'learning_rate': hp.uniform( 'learning_rate', 9e-6, 2e-3),
	'weight_decay': hp.uniform( 'weight_decay', 9e-5, 3e-3)
}


def get_params_VAE(args):

	params = sample( space_VAE )
	params = handle_integers( params )
	params['train_folder'] = args.train_folder
	params['batchsize'] = args.batchsize
	params['seed'] = args.seed
	params['cuda'] = args.cuda
	params['load_model'] = args.load_model
	params['model_type'] = args.model_type
	params['time_gap'] = 1
	params['num_images'] = 1
	params['stat_data_file'] = args.stat_data_file
	return params
