import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.UNet import UNet, NestedUNet
from utils.dataset import DrishtiDataset
from utils.DiceLoss import DiceLoss
from utils.logger import Logger

def train(net, device, dataset, batch_size=2, epochs=50, lr=1e-4):
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.RMSprop(params=net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = DiceLoss()

    log_train = Logger('log_train.txt')
    best_loss = float('inf')

    for epoch in range(epochs):
        train_loss = 0.0
        print('running epoch: {}'.format(epoch))
        net.train()
        for image, mask in tqdm(train_loader):
            image = image.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32)

            pred = net(image)
            loss = criterion(pred, mask)
            train_loss += loss.item() * image.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / len(train_loader.dataset)
        print('\tTraining Loss: {:.6f}'.format(train_loss))
        log_train.write_line(str(epoch) + ' ' + str(round(train_loss, 6)))

        if train_loss <= best_loss:
            best_loss = train_loss
            torch.save(net.state_dict(), 'model.pth')
            print('model saved')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    net = NestedUNet(n_channels=3, n_classes=1)
    # net = UNet(n_channels=3, n_classes=1)
    if os.path.isfile('model.pth'):
        net.load_state_dict(torch.load('model.pth', map_location=device))
    net.to(device=device)

    data_dir = 'data/train'
    image_transforms = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ])
    dataset_train = DrishtiDataset(data_dir=data_dir, image_transforms=image_transforms)
    train(net=net, device=device, dataset=dataset_train)

if __name__ == '__main__':
    main()