import os
import time
import json

import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import fcn_resnet50


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def batch_pre(weights_path, folder_path, palette_path, classes):
    aux = False  # inference time not need aux_classifier
    filename = os.listdir(folder_path)
    for i in range(len(filename)):
        img_path = folder_path + filename[i]

        assert os.path.exists(weights_path), f"weights {weights_path} not found."
        assert os.path.exists(img_path), f"image {img_path} not found."
        assert os.path.exists(palette_path), f"palette {palette_path} not found."
        with open(palette_path, "rb") as f:
            palette_dict = json.load(f)
            palette = []
            for v in palette_dict.values():
                palette += v

        # get devices
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(device))

        # create model
        model = fcn_resnet50(aux=aux, num_classes=classes + 1)

        # delete weights about aux_classifier
        weights_dict = torch.load(weights_path, map_location='cpu')['model']
        for k in list(weights_dict.keys()):
            if "aux" in k:
                del weights_dict[k]

        # load weights
        model.load_state_dict(weights_dict)
        model.to(device)

        # load image
        original_img = Image.open(img_path)

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.Resize(420),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225))])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            print("inference time: {}".format(t_end - t_start))

            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            mask = Image.fromarray(prediction)
            mask.save('D:/myProjects/evaluate/test_img_c/Prediction/fcn/' + filename[i].split('.')[0] + '.png')

            mask.putpalette(palette)
            mask.save('D:/myProjects/evaluate/test_img_c/Prediction_colored/fcn/' + filename[i].split('.')[0] + '.png')


def single_pre(weights_path, img_path, palette_path, classes):
    aux = False  # inference time not need aux_classifier

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."

    with open(palette_path, "rb") as f:
        palette_dict = json.load(f)
        palette = []
        for v in palette_dict.values():
            palette += v

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = fcn_resnet50(aux=aux, num_classes=classes + 1)

    # delete weights about aux_classifier
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    original_img = Image.open(img_path)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.Resize(420),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = Image.fromarray(prediction)

        mask.putpalette(palette)
        img = np.array(mask)

        mask.save('1.png')


if __name__ == '__main__':
    weightspath = "./save_weights/11-30-23_best_model.pth"

    folderpath = "D:/myProjects/evaluate/test_img_m/JPEGImages/"
    palettepath = "./palette.json"
    # roimask_path = "D:/myProjects/evaluate/test_img/unet_roi/"

    imgpath = 'data/mgs/JPEGImages/001ll.jpg'

    a = 2
    # batch_pre(weights_path=weightspath, folder_path=folderpath, palette_path=palettepath, classes=a)
    single_pre(weights_path=weightspath, img_path=imgpath, palette_path=palettepath, classes=a)
