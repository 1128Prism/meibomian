import json
import os
import time

import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def single_pre():
    classes = 2  # exclude background
    weights_path = "./save_weights/20231201-20_best_model.pth"
    img_path = "./data/mgs/test/images/100ll.jpg"
    roi_mask_path = "./data/mgs/test/mask/100llm.jpg"
    palette_path = "./palette.json"

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    palette = {
        "0": [0, 0, 0],  # 类别0对应黑色
        "1": [0, 255, 0],  # 类别1对应绿色
        "2": [0, 0, 255]  # 类别2对应蓝色
    }

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = UNet(in_channels=3, num_classes=classes + 1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
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
        # 预测单目标
        # # 将前景对应的像素值改成255(白色)
        # prediction[prediction == 1] = 255
        # # 将不敢兴趣的区域像素设置成0(黑色)
        # prediction[roi_img == 0] = 0
        # mask = Image.fromarray(prediction)
        # mask.save("test_result.png")

        # 预测多目标
        mask = Image.fromarray(prediction)
        palette_bytes = bytes(sum(palette.values(), []))
        # 将调色板应用到图像上
        mask.putpalette(palette_bytes)
        img = np.array(mask)
        cv2.imshow('1', img)
        cv2.waitKey(0)
        mask.save("test_result.png")


def batch_pre(weights_path, folder_path, palette_path, roi_mask, classes):
    filename = os.listdir(folder_path)
    for i in range(len(filename)):
        img_path = folder_path + filename[i]
        roi_mask_path = roi_mask + filename[i].split('.')[0] + 'c.png'

        assert os.path.exists(weights_path), f"weights {weights_path} not found."
        assert os.path.exists(img_path), f"image {img_path} not found."
        assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

        with open(palette_path, "rb") as f:
            palette_dict = json.load(f)
            palette = []
            for v in palette_dict.values():
                palette += v

        mean = (0.709, 0.381, 0.224)
        std = (0.127, 0.079, 0.043)

        # get devices
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using {} device.".format(device))

        # create model
        model = UNet(in_channels=3, num_classes=classes + 1, base_c=32)

        # load weights
        model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
        model.to(device)

        # load roi mask
        roi_img = Image.open(roi_mask_path).convert('L')
        roi_img = np.array(roi_img)

        # load image
        original_img = Image.open(img_path).convert('RGB')

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
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
            # 预测单目标
            # # 将前景对应的像素值改成255(白色)
            # prediction[prediction == 1] = 255
            # # 将不敢兴趣的区域像素设置成0(黑色)
            # prediction[roi_img == 0] = 0
            # mask = Image.fromarray(prediction)
            # mask.save("test_result.png")

            # 预测多目标
            mask = Image.fromarray(prediction)
            mask.save('D:/myProjects/evaluate/test_img_m/Prediction/unet/' + filename[i].split('.')[0] + '.png')

            mask.putpalette(palette)
            mask.save('D:/myProjects/evaluate/test_img_m/Prediction_colored/unet/' + filename[i].split('.')[0] + '.png')


if __name__ == '__main__':
    weightspath = "./save_weights/20231201-20_best_model.pth"

    folderpath = "D:/myProjects/evaluate/test_img_m/JPEGImages/"
    palettepath = "./palette.json"
    # roimask_path = "D:/myProjects/evaluate/test_img/unet_roi/"
    roimask_path = "D:/myProjects/evaluate/test_img_c/unet_roi/"

    a = 2
    # batch_pre(weights_path=weightspath, folder_path=folderpath, palette_path=palettepath, roi_mask=roimask_path,
    #           classes=a)

    single_pre()
