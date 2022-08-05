import numpy as np
import cv2
import torch
import glob
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# from model import create_model

#set the computation device
device = 'cpu' #torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#load the model and the trained weights
# model = create_model(num_classes=5).to(device)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#get no of input_features
in_features = model.roi_heads.box_predictor.cls_score.in_features
#define a new head for the detector with rwquired number of classes
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)
model.load_state_dict(torch.load('/home/sandbox-2/Documents/Gopal_office_file/Object_detection/microcontroller/outputs/model_100.pth', 
    map_location={'cuda:0': 'cpu'}))
model.eval()

#directory where all the images present
DIR_TEST = '/home/sandbox-2/Documents/Gopal_office_file/Object_detection/microcontroller/Microcontroller Detection/test'
test_images = glob.glob(f"{DIR_TEST}/*.jpg")
print(f"Test instances: {len(test_images)}")

#classes: 0 index is reserved for background
CLASSES = ['background', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora']

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.4

for i in range(len(test_images)):
    #get the img file to save later
    image_name = test_images[i].split('/')[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    #BGR 2 RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    #make the pixel range bwt 0 to 1
    image /= 255.0
    #bring color channels to front
    image = np.transpose(image,(2,0,1)).astype(float)
    #convert to tensor
    image = torch.tensor(image, dtype=torch.float) #.cuda()
    #add batch dimension
    image= torch.unsqueeze(image,0)
    with torch.no_grad():
        outputs = model(image)
    #load all detection to CPU for further operations
    outputs = [{k:v.to('cpu')for k,v in t.items()} for t in outputs]
    #carry further if there are detected boxes
    print(len(outputs))
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        #filter out boxes according to 'detection threshold'
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        print(boxes)
        print(scores)
        #get all the predicted class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

        #draw bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(
                orig_image,
                (int(box[0]),int(box[1])),
                (int(box[2]),int(box[3])),
                (0,0,255),2)
            cv2.putText(
                orig_image+'--'+scores, pred_classes[j],
                (int(box[0]),int(box[1]-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),
                2, lineType=cv2.LINE_AA)
        cv2.imshow('Prediction', orig_image)
        cv2.waitKey(0)
        cv2.imwrite(f"/home/sandbox-2/Documents/Gopal_office_file/Object_detection/microcontroller/test_predictions/{image_name}.png", orig_image,)
    print('Test predictions complete')
    cv2.destroyAllWindows()