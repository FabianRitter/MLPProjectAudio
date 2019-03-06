import sys
import os

import mlp.data_providers as data_providers
import numpy as np
from mlp.pytorch_experiment_scripts.arg_extractor import get_args
from mlp.pytorch_experiment_scripts.experiment_builder import ExperimentBuilder
from mlp.pytorch_experiment_scripts.model_architectures import ConvolutionalNetwork
import torch


args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed) # sets pytorch's seed

# convert number of filters per each layer to a list, and assert 
# it is specified for each layer

num_filters = [int(filt) for filt in args.num_filters[0].split(",")]

assert len(num_filters) == args.num_layers, "Not specified number of filter per each layer!"
print(" first assert!!!! now reading data")

train_data = data_providers.AudioDataProvider('valid', batch_size=args.batch_size,
                                               rng=rng,shuffle_order=False)  # initialize our rngs using the argument set seed
print("train read")
val_data = data_providers.AudioDataProvider('valid', batch_size=args.batch_size,
                                             rng=rng,shuffle_order=False)  # initialize our rngs using the argument set seed
print("val ok")
test_data = data_providers.AudioDataProvider('valid', batch_size=args.batch_size,
                                              rng=rng,shuffle_order=False)  # initialize our rngs using the argument set seed
print("test ok")


assert train_data.dict_ == val_data.dict_ == test_data.dict_, "Different dictionaries!"

print("second assert!!!")

custom_conv_net = ConvolutionalNetwork(  # initialize our network object, in this case a ConvNet
    input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_width),
    dim_reduction_type=args.dim_reduction_type,
        num_output_classes=train_data.num_classes, num_filters=num_filters,kernel_size = args.kernel_size,        num_layers=args.num_layers, use_bias=False)
print("definicion convolutional network ok")
conv_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data, val_data=val_data,
                                    test_data=test_data, batch_size = args.batch_size,
                                    training_instances = args.training_instances,
                                    test_instances = args.test_instances,
                                    val_instances = args.val_instances,
                                    image_height = args.image_height,
                                    image_width=args.image_width,
                                    use_cluster = args.use_cluster,
                                    gpu_id=args.gpu_id,
                                    args = args)  # build an experiment object
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
f=open("/disk/scratch/s1870525/datasets/experiment_config.txt", "a+")
f.write("network caracteristics")
f.write("num epochs %s \n",  args.num_epochs)
f.write("weight decay %s \n" , args.weight_decay_coefficients)
fwrite("kernel size %s \n", args.kernel_size)
fwrite("num layers %s \n" ,  args.num_layers)
fwrite("num filters %s \n" ,  args.num_filters)
f.close()
