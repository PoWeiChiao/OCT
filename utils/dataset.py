import glob
import os
import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DrishtiDataset(Dataset):
    def __init__(self, data_dir, image_transforms, is_random=True):
        self.data_dir = data_dir
        self.image_transforms = image_transforms
        self.is_random = is_random

        self.images_list = glob.glob(os.path.join(data_dir, 'image', '*.png'))
        self.masks_list = glob.glob(os.path.join(data_dir, 'mask', '*.png'))

        self.images_list.sort()
        self.masks_list.sort()

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = Image.open(self.images_list[idx])
        mask = Image.open(self.masks_list[idx])
        
        if self.is_random:
            isFlipLR = random.random()
            isFlipTB = random.random()
            if isFlipLR > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            if isFlipTB > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        mask = mask.resize((512, 512), 0)
        mask = np.array(mask)
        mask = np.where(mask > 0, 1, 0)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)

        image = self.image_transforms(image)

        return image, mask

def main():
    data_dir = 'D:/pytorch/Segmentation/Drishti/data/train'
    image_transforms = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ])
    dataset = DrishtiDataset(data_dir=data_dir, image_transforms=image_transforms)
    print(dataset.__len__())
    image, mask = dataset.__getitem__(0)
    print(image.shape)
    print(mask.shape)

if __name__ == '__main__':
    main()
        
