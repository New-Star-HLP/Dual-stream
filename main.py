'''
author:new-star
time:8.4.2023
'''
from typing import Any
import torch
import os
import math
import argparse
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.optim.lr_scheduler as lr_schedualer
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from vit_model import Vit
from torchvision import models
from FOV import FOV,reference_points_
from train_utils import train_one_epoch, evaluate,reinforce_train
from torchvision import transforms
import torch.nn.functional as F
import cub_200_2011 as dataset

class Res(nn.Module):
    def __init__(self):
        super().__init__()
        self.res=models.resnet50(pretrained=True)#nn.Identity()#
        self.fc=nn.Linear(1000,200)
    def forward(self,x):
        return self.fc(self.res(x))

class Net(nn.Module):
    def __init__(self,):
        super(Net,self).__init__()
        self.vit=Res()
        self.fc=nn.Linear(200,4)
        #weights= "./weights/base_model.pth" #'/home/omnisky/Downloads/HLP_code/exp_5/vit_base_patch16_224.pth' #
        #self.vit.load_state_dict(torch.load(weights,map_location="cuda"),strict=False) #)
        #self.vit.requires_grad_(False)
        self.fov=FOV(320,(320,1),(320,1))
        #self.compensate=nn.Parameter(torch.randn((2)))#torch.tensor([0.1,0.1]))#
    def forward(self,x,st=0):
        b,c,h,w=x.size()
        device,dtype=x.device,x.dtype
        #compensate=torch.tanh(self.compensate).mul(0.1)
        x1=self.fov(x)
        pos=reference_points_(b,h,x1[:,2],w,x1[:,3],device,dtype)
        
        x2=(x1[:,:2].reshape(-1,1,1,2)+pos)
        x_=F.grid_sample(  #grad:(-1,1)
                input=x, grid=x2[..., (1, 0)], mode='bilinear', align_corners=True)
        #if st>0: 
            #print(x1[0,])#,'\n',compensate
            #plt.imshow(x_[0,...].cpu().detach().T)
            #plt.show()
        return   self.vit(x_)
'''class Net(nn.Module):
    def __init__(self,):
        super(Net,self).__init__()
        self.vit=Vit()
        self.res=Res()
        self.fc=nn.Linear(200,200)
        weights= "./weights/base_model.pth" #'/home/omnisky/Downloads/HLP_code/exp_5/vit_base_patch16_224.pth' #
        self.vit.load_state_dict(torch.load(weights,map_location="cuda"),strict=False) #)
        #self.vit.requires_grad_(False)
        self.fov=FOV(320,(320,1),(320,1))
    def forward(self,x,st=0):
        b=x.size(0)
        device,dtype=x.device,x.dtype
        pos=reference_points(b,device,dtype)
        x1=self.fov(x)#+compensate
        x2=(x1.reshape(-1,1,1,2)+pos)
        x_=F.grid_sample(  #grad:(-1,1)
                input=x, grid=x2[..., (1, 0)], mode='bilinear', align_corners=True)
        #if st>0: 
            #print(x1[0,])
            #plt.imshow(x_[0,...].cpu().detach().T)
            #plt.show()
        x3=self.vit(x_)+self.fc(self.res(x))
        return  x3,x1   #self.vit(x_)  #'''

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(args)
    #tb_writer=SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    batch_size = args.batch_size
    train_loader=dataset.get_train_validation_data_loader((320,320),batch_size,None,True)
    val_loader=dataset.get_test_data_loader((320,320),batch_size,None,False)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print("using {} dataloader workers every process".format(nw))
    # #model1=reinforece_offset(320,(320,1),(320,1),10).to(device)  
    model =Net().to(device)
    #weights="./weights/base_model.pth"
    #model.load_state_dict(torch.load(weights,map_location="cuda"),strict=False)

    # 如果存在预训练权重则载入
    # if args.weights != "":
    #     if os.path.exists(args.weights):
    #         weights_dict = torch.load(args.weights, map_location=device)
    #         load_weights_dict = {k: v for k, v in weights_dict.items()
    #                              if model.state_dict()[k].numel() == v.numel()}
    #         print(model.load_state_dict(load_weights_dict, strict=False))
    #     else:
    #         raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    #if args.freeze_layers:
        #for name, para in model.named_parameters():
    #         # 除最后的全连接层外，其他权重全部冻结
            #if "vit"  in name:  #"fc"
                #para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad] #Adam(pg,lr=args.lr,weight_decay=1E-5)#
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    schedual = lr_schedualer.LambdaLR(optimizer, lr_lambda=lf)
    #stride=13
    for epoch in range(args.epochs):
        #if epoch<30:
        mean_loss = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                    device=device, epoch=epoch, warmup=True)
        #if epoch%5==4 or epoch>29:    
        #reinforce_train(model=model,optimizer=optimizer,stride=7+round(stride*(args.epochs-epoch)/args.epochs),data_loader=train_loader,
                            #device=device,epoch=epoch,warmup=True)
        
        schedual.step()
        # validate
        acc = evaluate(model=model, data_loader=val_loader, device=device)
        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags=["loss","accuracy","learning_rate"]
        #tb_writer.add_scalar(tags[0],mean_loss,epoch)
        #tb_writer.add_scalar(tags[1],acc,epoch)
        #tb_writer.add_scalar(tags[2],optimizer.param_groups[0]["lr"],epoch)
        torch.save(model.state_dict(),'./weights/FOV_model_dual.pth')
        # torch.save(model.state_dict(),"./weights/model-{}.path".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.00001)
    # dataset path
    #parser.add_argument('--data_path', type=str, default="E:\deeplearning\data\mini-imagenet")
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)  # False
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0,1 or cpu')  # 'cuda'

    opt = parser.parse_args()
    main(opt)


