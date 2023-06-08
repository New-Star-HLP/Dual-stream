import sys
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from FOV import reinforce_offset

from distributed_utils import reduce_value, is_main_process, warmup_lr_scheduler

def reinforce_train(model,optimizer,stride,data_loader,device,epoch,warmup):
    model.train()
    loss_1=nn.CrossEntropyLoss()
    loss_2=nn.MSELoss()  #nn.L1Loss() #
    optimizer.zero_grad()
    loss_sum=0.0
    loss1_=0.0
    loss2_=0.0
    if is_main_process():
        data_loader=tqdm(data_loader,file=sys.stdout)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    enable_amp = False and "cuda" in device.type
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    for step,data in enumerate(data_loader):
        _,images,labels=data
        
        pred_of=reinforce_offset(model,stride,images.to(device),labels.to(device))#.detach()
        with torch.cuda.amp.autocast(enabled=enable_amp):
            pred_la,offset=model(images.to(device))
            loss1=loss_1(pred_la,labels.to(device))#+loss_2(offset,pred_of)*16
            loss2=loss_2(offset,pred_of)*160+loss1
           #loss2.item()+loss1.item()
        #if loss.item()>0.5:
        scaler.scale(loss2).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        #loss2.backward(retain_graph=True) #
        
        loss_sum+=loss1.item()
        loss1_+=loss1.item()#-loss2.item())
        loss2_+=loss2.item()
        #loss2.backward()
        #optimizer.step()
        #optimizer.zero_grad()
        if is_main_process():
            info="[epoch {}],loss_1: {:.3f},loss_2:{:.3f},lr:{:.5f},stride:{:d}".format(
                epoch,loss1_/(step+1),
                loss2_/(step+1),optimizer.param_groups[0]["lr"],stride)
            data_loader.desc = info
        if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
            lr_scheduler.step()
    print('loss:{:.3f}'.format(loss_sum/(step+1)))

def train_one_epoch(model, optimizer, data_loader, device, epoch, use_amp=False, warmup=True):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    enable_amp = use_amp and "cuda" in device.type
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)

    sample_num = 0
    for step, data in enumerate(data_loader):
        _,images, labels = data
        sample_num += images.shape[0]

        with torch.cuda.amp.autocast(enabled=enable_amp):
            pred,_= model(images.to(device))
            loss = loss_function(pred, labels.to(device))

            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss = reduce_value(loss, average=True)
        accu_loss += loss.detach()

        # 在进程0中打印平均loss
        if is_main_process():
            info = "[epoch {}] loss: {:.3f}, train_acc: {:.3f}, lr: {:.5f}".format(
                epoch,
                accu_loss.item() / (step + 1),
                accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"])
            data_loader.desc = info

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
            lr_scheduler.step()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 验证集样本个数
    num_samples = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
    st=20
    for step, data in enumerate(data_loader):
        _,images, labels = data
        pred,_= model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()
        st-=1
    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)
    acc = sum_num.item() / num_samples

    return acc

"""
class My_loss(nn.Module):  #restrain/constrain the classes probability
    def __init__(self):
        super(My_loss, self).__init__()
        self.l1=nn.L1Loss()
        self.l2=nn.CrossEntropyLoss()
    def forward(self,pred_y,labels): #b num_classes
        b,_=pred_y.size()
        device='cuda' if torch.cuda.is_available() else 'cup'
        x=torch.sum(pred_y,dim=1,keepdim=True).to(device)
        #x1=torch.mean(pred_y,dim=1,keepdim=True).to(device)
        m=torch.ones((b,1)).to(device)
        loss1=self.l1(m,x)*0.5
        loss2=self.l2(pred_y,labels)
        #loss2=self.l1(torch.ones((b,1))*1.0/b,x1)
        return loss1+loss2
"""