'''
mutil-heads focus detection
multil-sacle feature-imgs, but fc_layer is fixed
'''
import torch
import torch.nn as nn

class Focus_detection_deformable(nn.Module):  #
    def __init__(self):
        super(Focus_detection_deformable, self).__init__()
        self.offset_heads=nn.Sequential()
    def forward(self,q_vector):
        pass

class Attn_(nn.Module):
    def __init__(self,dim):
        super(Attn_, self).__init__()
        self.scale=dim**-0.5
        self.v=nn.Conv1d(dim,dim,1,1)
        self.k=nn.Conv1d(dim,dim,1,1)
    def forward(self,x):
        k=self.k(x)
        v=self.v(x)
        attn=torch.matmul((x@k.transpose(2,1))*self.scale,v)
        return attn
class Block(nn.Module):
    def __init__(self,in_dim,out_dim,depth):
        super(Block, self).__init__()
        self.block=nn.Sequential(*[nn.LayerNorm(in_dim[i]),Attn_(i),
                                   nn.Conv1d(in_dim[i],out_dim[i],1,1)] for i in range(depth))
    def forward(self,x):
        return self.block(x)
class Mulit_scale(nn.Module):  #can process the mutil-scale input imgs   ??
    def __init__(self,in_dim,out_dim,kernel_size,stirde):
        super(Mulit_scale, self).__init__()
        self.patch_=nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size,stirde),nn.GELU())
        self.block=Block(in_dim=[197,197],out_dim=[197,197],depth=2)
        self.cls_token=nn.Parameter(torch.randn(1,1))

    def forward(self,imgs):
        x=self.patch_(imgs)
        b,c,n=x.size()
        cls_token=self.cls_token[None,...].expand(b,1,n)
        x1=torch.stack((cls_token,x),dim=-1)
        x2=self.block(x1)

