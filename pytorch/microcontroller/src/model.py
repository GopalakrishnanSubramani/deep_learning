import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes):
    #load faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    #get no of input_features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    #define a new head for the detector with rwquired number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
