import cv2 as cv
import glob
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model.UNet import UNet, NestedUNet
from utils.dataset import DrishtiDataset

def save(save_dir, image):
    cv.imwrite(save_dir, image)

def create_label_on_image(image_dir, label_dir):
    image = Image.open(image_dir)
    image = image.resize((512, 512))
    label = Image.open(label_dir)

    image = np.array(image)
    image = image[:, :, ::-1]

    label_blue = np.zeros((512, 512))
    label_green = np.array(label)
    label_red = np.zeros((512, 512))

    mask = np.stack([label_blue, label_green, label_red], axis=-1)
    mask = np.array(mask, dtype=np.uint8)

    image = cv.addWeighted(image, 1, mask, 0.3, 0)
    save(os.path.join('predict_mask', os.path.basename(image_dir)), image)

def predict(net, device, dataset, test_dir):
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    output = []
    net.eval()
    with torch.no_grad():
        for i, (image, mask) in enumerate(test_loader):
            image = image.to(device=device, dtype=torch.float32)
            pred = net(image)
            pred = np.array(pred.data.cpu()[0])[0]
            pred = np.where(pred > 0.5, 255, 0)
            pred = np.array(pred, dtype=np.uint8)
            output.append(pred)
    return output

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = NestedUNet(n_channels=3, n_classes=1)
    # net = UNet(n_channels=3, n_classes=1)
    net.load_state_dict(torch.load('model.pth', map_location=device))
    net.to(device=device)

    data_dir = 'data/test'
    image_transforms = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ])
    dataset_test = DrishtiDataset(data_dir=data_dir, image_transforms=image_transforms, is_random=False)
    test_dir = 'data/test'
    test_list = glob.glob(os.path.join(test_dir, 'image', '*.png'))
    test_list.sort()
    pred = predict(net, device, dataset_test, test_dir)
    for i, (test) in enumerate(test_list):
        save(os.path.join('predict', os.path.basename(test)), pred[i])
        create_label_on_image(os.path.join(test_dir, 'image', os.path.basename(test)), os.path.join('predict', os.path.basename(test)))
        
if __name__ == '__main__':
    main()

