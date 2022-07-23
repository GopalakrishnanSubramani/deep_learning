import torch

import pandas as pd
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as T
import config
import cv2
import os

#Training transforms
# def image_transforms():
transforms = T.Compose(
        [        T.Resize([config.IMAGE_SIZE,config.IMAGE_SIZE]),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.25,0.25,0.25]),
        ]                                         
    )
    # return transforms

class DogCatDataset(Dataset):
  def __init__ (self, ROOT_DIR, CSV_DIR, transform=transforms,target_transform=None):
    self.img_labels = pd.read_csv(CSV_DIR, sep=(','))
    self.img_dir = ROOT_DIR
    self.transform = transform
    self.target_transform = target_transform
  def __len__(self):
    return len(self.img_labels)
  def __getitem__(self, index):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[index,0])
    image = cv2.imread(img_path)
    label = self.img_labels.iloc[index,1]
    
    if self.transform == transforms:
            image = self.transform(image)
    return image,label

def get_datasets(ROOT_DIR, CSV_DIR,BATCH_SIZE,NUM_WORKERS):
    dataset = DogCatDataset(ROOT_DIR,CSV_DIR)
    dataset_size = len(dataset)
    print(f"total no of images{dataset_size}")

    #Training and validation sets
    train_per = int(len(dataset)*0.65)
    val_per = int(len(dataset)*0.25)
    test_per = len(dataset)-train_per-val_per

    train_data, validation_data, test_data= torch.utils.data.random_split(dataset,[train_per,val_per,test_per])

    train_loader = torch.utils.data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True, num_workers=NUM_WORKERS)
    validation_loader = torch.utils.data.DataLoader(validation_data,batch_size=BATCH_SIZE,shuffle=True, num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)

    print(f"Total training images: {len(train_data)}")
    print(f"Total validation images: {len(validation_data)}")

    return train_loader, validation_loader, test_loader #, train_data

if __name__ == '__main__':
    dogdata = get_datasets(config.DATA_ROOT_DIR,config.CSV_DIR,4,2)
    img = dogdata[3]
    # img = train_data[0][1]
    img1 = img[0][0]
    img1 = T.ToPILImage()(img1).convert("RGB")
    img1.show()
    pass