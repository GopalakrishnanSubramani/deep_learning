from email.quoprimime import body_encode
import cv2
import numpy as np
import random
import torch
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def get_outputs(image, model, threshold):
    with torch.no_grad():
        #forward pass of the image through model
        outputs = model(image)

    #get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    #index of the scores above the threshold
    threshold_preds_indicies = [scores.index(i) for i in scores if i > threshold]
    threshold_preds_count = len(threshold_preds_indicies)
    #get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    #avoid masks beloe the threshold
    masks = masks[:threshold_preds_count]

    #get the bounding boxes, (x1,y1)(x2,y2) format
    boxes = [[(int(i[0]),int(i[1])),(int(i[2]),int(i[3]))] for i in outputs[0]['boxes']]
    #discord the bounding boxes below threshold
    boxes = boxes[:threshold_preds_count]

    #get the class labels
    labels = [coco_names[i] for i in outputs[0]['labels']]
    return masks, boxes, labels

def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1
    beta = 0.6
    gamma = 0

    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i].astype(np.uint8))
        green_map = np.zeros_like(masks[i].astype(np.uint8))
        blue_map = np.zeros_like(masks[i].astype(np.uint8))    

        #apply a random color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i]]==1, green_map[masks[i]]==1,blue_map[masks[i]]==1
        #combine all masks into the single image
        segmentation_map = np.stack([red_map, green_map,blue_map], axis=2)
        #convert the original PIL image into NumPy format
        image = np.array(image)
        #convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #apply mask on the image
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)

        #draw bounding boxes
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, thickness=2)

        #put the label text above the objects
        cv2.putText(image, labels[i],(boxes[i][0][0], boxes[i][0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2, lineType=cv2.LINE_AA)

        return image