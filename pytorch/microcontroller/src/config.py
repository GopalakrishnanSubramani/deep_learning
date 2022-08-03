import torch

BATCH_SIZE = 4 #increase / decrease according to GPU memeory
RESIZE_TO = 512 #resize the image for training and transforms
NUM_EPOCHS = 100 #number of epochs to train for

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#training images and xml files dir
TRAIN_DIR = '/home/sandbox-2/Documents/Gopal_office_file/Object_detection/microcontroller/Microcontroller Detection/train'

#validation images and xml files dir
VALID_DIR = '/home/sandbox-2/Documents/Gopal_office_file/Object_detection/microcontroller/Microcontroller Detection/test'

CLASSES = ['background', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora']

NUM_CLASSES = 5

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

OUT_DIR = '/home/sandbox-2/Documents/Gopal_office_file/Object_detection/microcontroller/outputs'
SAVE_PLOTS_EPOCH =2
SAVE_MODEL_EPOCH =2
