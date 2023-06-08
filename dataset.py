import os
import json
from PIL import Image
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import GradCAM,show_cam_on_image
from torchvision import transforms
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    """自定义数据集"""
    def __init__(self,
                 root_dir: str,
                 csv_name: str,
                 json_path: str,
                 transform=None):
        images_dir = os.path.join(root_dir, "images")
        assert os.path.exists(images_dir), "dir:'{}' not found.".format(images_dir)

        assert os.path.exists(json_path), "file:'{}' not found.".format(json_path)
        self.label_dict = json.load(open(json_path, "r"))

        csv_path = os.path.join(root_dir, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found.".format(csv_path)
        csv_data = pd.read_csv(csv_path)
        self.total_num = csv_data.shape[0]
        self.img_paths = [os.path.join(images_dir, i)for i in csv_data["filename"].values]
        self.img_label = [self.label_dict[i][0] for i in csv_data["label"].values]
        self.labels = set(csv_data["label"].values)

        self.transform = transform

    def __len__(self):
        return self.total_num

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.img_paths[item]))
        label = self.img_label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
class Reshape_Transform:  #change the output->[b c h w] type
    def __init__(self,model):
        pass
    def __call__(self, x):
        pass
def display_cam(model,model_patch,target_layer,img_path,img_liabel):
    model=model   #choose model
    weights_path=model_patch
    model.load_state_dict(torch.load(weights_path,map_location="cuda" if torch.cuda.is_available() else "cpu"))
    target_layer=target_layer  #choose a layer of model ,but the layer should be feature_map layer
    data_transformer=transforms.Compose([transforms.ToTensor(),])
    img_path=img_path
    assert os.path.exists(img_path),"file:'{}' does not exist.".format(img_path)
    img=Image.open(img_path).convert('RGB')
    img=np.array(img,dtype=np.int8)
    #img=center_crop_img(img,224)  #crop the img according with the size
    img_tensor=data_transformer(img)  #[c h w]
    input_tensor=torch.unsqueeze(img_tensor,dim=0) #[1 c h w]

    cam=GradCAM(model=model,target_layer=target_layer,use_cuda=False,reshape_transform=Reshape_Transform)
    target_category=img_liabel
    gray_scale_cam=cam(input_tensor,target_category)
    gray_scale_cam=gray_scale_cam[0,:]
    visualization=show_cam_on_image(img/255,gray_scale_cam,use_rgb=True)
    plt.imshow(visualization)
    plt.savefig('./visualization/1')
    plt.show()



