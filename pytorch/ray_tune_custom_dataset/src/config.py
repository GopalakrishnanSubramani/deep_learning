import os

#training parameters
EPOCHS = 10

#Data root
DATA_ROOT_DIR = os.path.abspath('/home/krish/Documents/dogs-vs-cats/ray_tune_custom/input/dataset/')
CSV_DIR = os.path.abspath('/home/krish/Documents/dogs-vs-cats/ray_tune_custom/input/dataset.csv')


# images='/home/krish/Documents/dogs-vs-cats/dataset/dataset_3000img'
# label="/home/krish/Documents/dogs-vs-cats/dataset/dataset3000.csv"

#number of works
NUM_WORKERS = 4

#Ratio of split to use for validation
VALID_SPLIT = 0.1

# Image to resize to in tranforms.
IMAGE_SIZE = 224

# For ASHA scheduler in Ray Tune
MAX_NUM_EPOCHS = 50
GRACE_PERIOD = 1

# For search run (Ray Tune settings).
CPU=1
GPU=1

# Number of random search experiments to run.
NUM_SAMPLES=20
