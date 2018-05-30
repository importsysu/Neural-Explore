import os

import tensorflow as tf

flags = tf.app.flags

# Names and directories
flags.DEFINE_string("exp_name", "basic", "Model name [basic]")
flags.DEFINE_string("out_base_dir", "out", "out base dir [out]")

# Device placement
flags.DEFINE_string("device", "/cpu:0", "default device for summing gradients. [/cpu:0]")
flags.DEFINE_string("device_type", "gpu", "device for computing gradients (parallelization). cpu | gpu [gpu]")
flags.DEFINE_integer("num_gpus", 1, "num of gpus or cpus for computing gradients [1]")

# Essential training and test options
flags.DEFINE_string("mode", "test", "trains | test | forward [test]")
flags.DEFINE_integer("validation_size", 5000, "validation size [5000]")

# Training / test parameters
flags.DEFINE_integer("batch_size", 64, "Batch size [64]")
flags.DEFINE_integer("num_epochs", 10, "Total number of epochs for training [10]")
flags.DEFINE_integer("num_steps", 20000, "Number of steps [20000]")
flags.DEFINE_integer("load_step", 0, "load step [0]")
flags.DEFINE_float("init_lr", 0.01, "Initial learning rate [0.01]")
flags.DEFINE_float("wd", 5e-4, "L2 weight decay for regularization [5e-4]")

# Optimizations
#flags.DEFINE_bool("cluster", False, "Cluster data for faster training [False]")

# Logging and saving options
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_integer("log_period", 100, "Log period [100]")
flags.DEFINE_integer("eval_period", 100, "Eval period [100]")
flags.DEFINE_integer("save_period", 1000, "Save Period [1000]")
flags.DEFINE_integer("max_to_keep", 20, "Max recent saves to keep [20]")
flags.DEFINE_float("decay", 0.9, "Exponential moving average decay for logging values [0.9]")

#Model parameters
flags.DEFINE_boolean("use_placeholder", True, "use placeholder? [True]")