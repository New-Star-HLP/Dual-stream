import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_model import vit_base_patch16_224
from torchvision import transforms
class Patch_(nn.Module):
    def __init__(self,dim,patch,stride):
        super(Patch_, self).__init__()
        self.proj_ = nn.Conv2d(3, dim, patch, stride)
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, dim)
    def forward(self, imgs):
        x = self.proj_(imgs).flatten(2).transpose(1,2)
        x = self.norm(self.proj_out(x)) #
        return x
@torch.no_grad()
def reference_points(b,device,dtype):
    x, y = torch.meshgrid(
        torch.linspace(-0.98, 0.4, 224, device=device, dtype=dtype),
        torch.linspace(-0.98, 0.4, 224, device=device, dtype=dtype)
    )
    ref = torch.stack((x, y), dim=-1)
    return ref[None, ...].expand(b, -1, -1, -1)   
 
class FOV(nn.Module):
    def __init__(self,dim,patch,stride):
        super(FOV, self).__init__()
        self.patch=Patch_(dim,patch,stride)
        self.mlp=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim),nn.LayerNorm(dim))
        self.cls=nn.Parameter(torch.zeros(1,1,dim))
        self.postion=nn.Parameter(torch.zeros(1,1,320))
        self.offset=nn.Linear(dim,4)
        self.fator=nn.Parameter(torch.tensor([0.1,0.1,0.1,0.1]))
        self.compensate=nn.Parameter(torch.randn(2))
        self.reset_par()
    def reset_par(self):
        for m in self.parameters():
            if isinstance(m,nn.Linear):
                nn.init.kaiming_normal_(m.weight,0.2)
                nn.init.xavier_normal_(m.bias,.5)
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight,0.3)
                nn.init.constant_(m.bias,0.1)

    def forward(self,x):
        b,c,h,w=x.size()
        q=self.patch(x)
        cls=self.cls.expand(x.size(0),-1,-1)
        q_=torch.cat((cls,q),dim=1)+self.postion
        q_=self.mlp(q_).pow(2)
        factor=self.fator     #.clamp(0.01,10.0)
        offset=torch.tanh(torch.relu(self.offset(q_)*factor))  #.max(dim=1)
        #scale=torch.tensor([h*0.6/320,w*0.6/320]).to("cuda")
        offset[:,2]=torch.round(offset[:,2]*h)
        offset[:,3]=torch.round(offset[:,3]*w)
        offset[:,:2]=offset[:,:2].mean(dim=1)#+torch.tanh(self.compensate).mul(0.05)#.clamp(0.,0.6)
        return offset
@torch.no_grad()
def reference_points_(b,h,h_,w,w_,device,dtype):
    m_h,m_w=2.0/h,2.0/w
    h_=int(h_.cpu().detach().numpy())
    h=h-h_
    w_=int(w_.cpu().detach().numpy())
    w=w-w_
    x, y = torch.meshgrid(
        torch.linspace(-0.98, -0.98+m_h*h, h, device=device, dtype=dtype),
        torch.linspace(-0.98, -0.98+m_w*w, w, device=device, dtype=dtype)
    )
    ref = torch.stack((x, y), dim=-1)
    return ref[None, ...].expand(b, -1, -1, -1) 

@torch.no_grad()
def reinforce_offset(model,stride,imgs,labels):
    b,c,h,w=imgs.size()
    device,dtype=imgs.device,imgs.dtype
    loss_fun=nn.CrossEntropyLoss(reduce=False)
    offset=model.fov(imgs)
    offset__=torch.zeros_like(offset).to(device)
    offset_= offset#torch.zeros((b, 2)).to(device)
    min_loss = torch.zeros((b, 1)).to(device)
    #h=224#160+round(64*(stride-7)/13)
    #w=224#160+round(64*(stride-7)/13)
    ref=reference_points(b,device=device,dtype=dtype)
    for i in [-1,0,1]:
        for j in [-1,1,2]:   
            offset__[...,0]=offset[...,0]+i*stride*1.0/160
            offset__[...,1]=offset[...,1]+j*stride*1.0/160
            pos=offset__.reshape(-1,1,1,2)+ref#
            #pos=torch.where(pos>-0.98,pos,-0.98)
            #pos=torch.where(pos<0.98,pos,0.98)
            x_ = F.grid_sample(
                input=imgs, grid=pos[..., (1, 0)], mode='bilinear', align_corners=True)
            pred_x = model.vit(x_)
            loss =loss_fun(pred_x, labels).data #torch.abs(pred_x-labels)*320#.data
            if i==-1 and j==-1:
                min_loss=loss
            for k in range(b):
                if min_loss[k] > loss[k]:
                    min_loss[k] = loss[k]
                    offset_[k, 0] = offset[k,0]+i*stride*1.0/160
                    offset_[k, 1] = offset[k,1]+j*stride*1.0/160    
    return offset_   






