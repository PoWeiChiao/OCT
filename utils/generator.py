import glob
import os
import shutil

def main():
    root = 'D:/pytorch/Segmentation/Drishti'
    raw_data_dir = 'Drishti-GS1_files'
    target = 'data'

    separator = ['Test', 'Training']
    for s in separator:
        images_list = glob.glob(os.path.join(root, raw_data_dir, s, 'Images', '*.png'))
        for image in images_list:
            ss = 'train'
            if s == 'Test':
                ss = 'test'
            shutil.copyfile(image, os.path.join(root, target, ss, 'image', os.path.basename(image)))
            masks_list = glob.glob(os.path.join(root, raw_data_dir, s, 'GT', os.path.basename(image)[:-4], 'SoftMap', '*.png'))
            for m in masks_list:
                if 'ODseg' in m:
                    shutil.copyfile(m, os.path.join(root, target, ss, 'mask', os.path.basename(image)))

if __name__ == '__main__':
    main()