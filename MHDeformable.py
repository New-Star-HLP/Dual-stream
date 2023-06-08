'''
/mutli/-heads deformable offset
'''
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module,Sequential,Conv2d,Linear,LayerNorm,Parameter,Conv1d
from einops import rearrange
from wave_mlp import PATM

class Patch_Embedding(Module):
    def __init__(self,dim,patch=(32,16),stride=(16,8)):
        super(Patch_Embedding, self).__init__()
        self.proj_=Conv2d(3,dim,patch,stride)
        self.norm=LayerNorm(dim)
        self.proj_out=Linear(dim,dim)
    def forward(self,imgs,**kwargs):
        x=self.proj_(imgs).flatten(2).transpose(1,2)
        if hasattr(self,"norm"):
            x=self.norm(self.proj_out(x))
        return x
class Q_vector(Module): #the Q_vector can share
    def __init__(self,dim):
        super(Q_vector, self).__init__()
        self.Q_vector=Linear(dim,dim)
    def forward(self,x):
        return self.Q_vector(x)
class Deformable_Focus_(Module):#视场不同会造成不一样的矩阵
    def __init__(self):
        super(Deformable_Focus_, self).__init__()
        self.q=nn.Sequential()
        self.offset=nn.Sequential()
    @torch.no_grad()
    def reference_point(self,s_x,s_y,device,dtype):
        x,y=torch.meshgrid(
            torch.linspace(-0.99,0.+s_x*1.0/160,160+s_x,device,dtype),
            torch.linspace(-0.99,0.+s_y*1.0/160,160+s_y,device,dtype)
        )
        ref=torch.stack((x,y),dim=-1)
        return ref      #X/Y: (-1,0.+S_)
    def forward(self,imgs):  #针对单个样本，会降低运行速度
        q_vector=self.q(imgs)
        b,c,n=q_vector.size()
        device,dtype=q_vector.device(),q_vector.dtype()
        offset=self.offset(q_vector)

        ref=self.reference_point(b,device,dtype)
class Deformable_(Module):#两种情况：可以有视场形状的变形
    def __init__(self,dim):
        super(Deformable_, self).__init__()
        self.patch_=Patch_Embedding(dim)
        self.q=Q_vector(dim)
        #self.offset_=nn.Sequential()  #b 2->b h w 2
        self.offset_row=nn.Sequential() #b 1->b h w 1
        self.row_column=nn.Sequential()  #b w 1 ->b h w 1
    @torch.no_grad()
    def reference_point(self, B, H, W, dtype, device):
        x, y = torch.meshgrid(
            torch.linspace(-112, 112, 224, device, dtype),
            torch.linspace(-112, 112, 224, device, dtype),)
        ref = torch.stack((x, y), dim=-1)
        ref[..., 0].div_(H // 2)
        ref[..., 1].div_(W // 2)
        return ref[None, ...].expand(B, -1, -1, -1)  # x/y:(-0.7,0.7)
    def forward(self,imgs):
        b,c,h,w=imgs.size()
        device,dtype=imgs.device,imgs.dtype
        q=self.q(self.patch_(imgs))
        #row_column=torch.relu(torch.sofmax(self.row_column(q))-o.5).mul(0.3)  #b w 1
        row_column=torch.tanh(self.row_column(q)).mul(0.3)
        row_column[-1,None,...].epxand(-1,h,-1,-1)
        offset_=torch.tanh(self.offset_row(q)).mul(0.3)
        offset_[-1,None,None,-1].expand(-1,h,w,-1)
        offset=torch.stack((offset_,row_column),dim=-1) #b h w 2
        ref=self.reference_point(b,h,w,device,dtype)
        pos=offset+ref
        x_ = F.grid_sample(  #
            input=imgs, grid=pos[..., (1, 0)], mode='bilinear', align_corners=True)
        return self.patch_embedding(x_)  # x_:b 3 224 224

class Deformable_offset(Module):#以点为单位，散且少约束，因此考虑采用整体块的偏移（类似于warpaffine
    #if offset:(b h w 2) ,so that's means the calculated amount is larger than (b n c)'s
    def __init__(self,dim,offset_dim=2):#以块偏移貌似不需要多头
        super(Deformable_offset, self).__init__()
        self.theta=Sequential()  #warpaffine may be diffict
        self.offset_linear=Sequential(Linear(dim,1),rearrange('b n c->b (n c)'),
                                      Linear(741,offset_dim))
        self.patch_embedding=Patch_Embedding(dim,patch=(32,16),stride=(16,8))
        self.q_vector=Q_vector(dim)
        self.reset_parameters() #init the parameters
    def reset_parameters(self):
        for m in self.parameters():
            if isinstance(m,(nn.Linear,nn.Conv2d)):
                nn.init.constant_(m.weights,0.)
                nn.init.constant_(m.bias,0.)
    @torch.no_grad()
    def reference_point(self,B,H,W,dtype,device):
        x,y=torch.meshgrid(
            torch.linspace(-112,112,224,device,dtype),
            torch.linspace(-112,112,224,device,dtype),
        )
        ref=torch.stack((x,y),dim=-1)
        ref[...,0].div_(H//2)
        ref[...,1].div_(W//2)
        return ref[None,...].expand(B,-1,-1,-1)  #x/y:(-0.7,0.7)
    def forward(self,imgs): #q_vector: b n c
        B,C,H,W=imgs.size()
        dtype,device=imgs.dtypr,imgs.device
        q_vector=self.q_vector(self.patch_embedding(imgs))
        torch.save(self.q_vector,'.\weights\Q_Vector.pth')
        reference=self.reference_point(B,H,W,dtype,device) #b h w 2
        #offset=self.offset_linear(q_vector).mul_(0.1).clamp_(0.,0.3)
        offset=torch.tanh(self.offset_linear(q_vector)*0.01).mul_(0.3) #b 2, and (-0.3,0.3)
        pos=reference+offset #广播到相同维度
        x_ = F.grid_sample(  #
            input=imgs, grid=pos[..., (1, 0)], mode='bilinear', align_corners=True)
        return self.patch_embedding(x_)  #x_:b 3 224 224
class Focus_deformable(Module):#在Q_vecotr内实现多头偏移,可用视场内的，也可以用整个目标来计算
    #视焦区域的头多（多种偏移量）,refere Deformable DERT,allow the points can offset by itself
    def __init__(self,dim,heads,):
        super(Focus_deformable, self).__init__()
        #self.patch_=Patch_Embedding(dim,(32,16),(16,8))
        #self.q_=Q_vector(dim)
        #assert os.path.exists('.\weights\Q_vector.pth'), 'the weights is not exists'
        #self.q_vector.load_state_dict(torch.load('.\weights\Q_vector.pth'),
                                      #map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
        #self.q_.requires_grad_(False)
        self.offset_focus=Sequential(Conv1d(dim,20,7,5),Linear(37,10))#
        self.attn=Linear(dim*heads,dim*heads)
        self.reset_parameters()
    def reset_parameters(self):
        for m in self.parameters():
            if isinstance(m,(nn.Linear,nn.Conv2d)):
                nn.init.constant_(m.weights,0.)
                nn.init.constant_(m.bias,0.)
    @torch.no_grad()
    def reference(self,B,N,C,device,dtype):#b 351 196
        x,y=torch.meshgrid(
            torch.linspace(0.5,20-0.5,28,device,dtype),
            torch.linspace(0.5,10-0.5,7,device,dtype)
        )
        ref=torch.stack((x,y),dim=-1)
        ref[...,0].div_(N)
        ref[...,1].div_(C)
        return ref[None,...].expand(B,-1,-1,-1)
    def forward(self,q_vector):#q:b n=351 c=196 offset:b n c*heads, attn:b n c*heads
        B,N,C=q_vector.size()
        device,dtype=q_vector.device,q_vector.dtype
        offset=torch.tanh(self.offset_focus(q_vector)).mul(0.9) #b 20 10 2
        ref=self.reference(B,N,C,device,dtype)
        pos=ref+offset
        x_=F.grid_sample(q_vector.reshape(-1,1,-1,-1),pos[...,(-1,0)],'linear',
                         align_corners=True)#.reshape(B,20,16)
        x_=rearrange(x_,'b c n h->b (c h) n')
        return x_
class Attn(Module):#
    def __init__(self,dim,ids):
        super(Attn, self).__init__()
        #self.patch_=Patch_Embedding(dim,(32,16),(16,8)) #the weights can share with the foremention
        self.q_vector=Q_vector(dim)  #the weights is shared by Deformable_offset
        if ids==0:
            assert os.path.exists('.\weights\Q_vector.pth'),'the weights is not exists'
            self.q_vector.load_state_dict(torch.load('.\weights\Q_vector.pth'),
                                          map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
            self.q_vector.requires_grad_(False)
        self.k_vector=Linear(dim,dim)
        self.v_vector=Linear(dim,dim)
        self.scale=dim**-0.5
    def forward(self,x): #x:b 351,196
        #x=self.patch_(x_)  #
        q_v=self.q_vector(x)
        k_v=self.k_vector(x)
        v_v=self.v_vector(x)
        attn=torch.matmul(torch.matmul(q_v,k_v.transpose(-1,-2))*self.scale,v_v)
        return attn

class Block(Module):#dim:list len(dim)==depth
    def __init__(self,dim,depth,):
        super(Block, self).__init__()
        self.cls_token=Parameter(torch.randn())
        self.pos_emb=Parameter(torch.linspace(0.1,19.6,196,))##
        assert len(dim)==depth, "the len of dim(list) is not matching with the depth"
        self.transformer=Sequential(*[LayerNorm(dim[i]),Attn(dim[i],i),Linear(dim[i],4*dim[i]),
                                      Linear(4*dim[i],dim[i]),LayerNorm(dim)] for i in range(depth))
    def forward(self,x):
        return self.transformer(x)

class Net(Module):#先要求取视场偏移量，后用新的视场范围来求attn及视焦的贡献度
    def __init__(self,dim,patch,stride):
        super(Net, self).__init__()
        self.patch_=Patch_Embedding(dim,patch,stride)
        self.q_=Q_vector(dim)
        self.deformable_offset=Deformable_offset(dim)
        self.focus_=Focus_deformable(dim,offset_dim=2,heads=4)
        self.transformer=Block(dim,depth=3)  #combine the foucs dim
        self.classfication=Linear(dim,100)
    def forward(self,imgs):
        x=self.deformable_offset(imgs)
        x=self.patch_(x)
        q=self.q_(x)
        focus=self.focus_(q).reshape(imgs.size(0),1,-1)
        x_=torch.cat((focus,q),dim=1)
        x1=self.transformer(x_)
        x1=x1[:,0]
        return self.classfication(x1)

class lstm_conv1d(Module):
    def __init__(self,in_dim,hidden_dim,out_dim,kernel_size_out,stride_out):# kernle_size_hidden,stride_hidden
        super(lstm_conv1d, self).__init__()
        self.in_dim=in_dim
        self.hidden_dim=hidden_dim
        self.out_dim=out_dim
        self.kernel_size1=kernel_size_out
        #self.kernel_size2=kernle_size_hidden
        self.stride1=stride_out
        #self.stride2=stride_hidden
        self.out_conv1d=Conv1d(in_channels=self.in_dim,out_channels=self.out_dim*4,
                           kernel_size=self.kernel_size1,stride=self.stride1)
        #self.hidden_conv1d=Conv1d(in_channels=self.hidden_dim,out_channels=self.out_dim*4,
                                  #kernel_size=self.kernel_size2,stride=self.stride2)
    def forward(self,x,cur=None): #x: b n c/b c n,cur:b n c/b c n
        #cur_c,cur_h=cur4
        cur_c=cur  #假设只需要上时刻的状态, the dim is identated with the x
        if cur_c is None:
            cur_c=torch.zeros_like(x)
        #assert cur_h.size(-1)==self.hidden_dim, 'the hidden_dim is not matching'
        x1=self.out_conv1d(x)
        #x1_=self.hidden_conv1d(cur_h)
        c_i,c_f,c_o,c_g=torch.splite(x1,4,dim=1)
        #c_i_,c_f_,c_o_,c_g_=torch.split(x1_,4,dim=1)
        i=torch.sigmoid(c_i)  #+c_i——
        f=torch.sigmoid(c_f)  #+c_f_
        o=torch.sigmoid(c_o) #+c_o_
        g=torch.tanh(c_g) #+c_g_
        next_c=f*cur_c+i*g
        next_h=o*torch.tanh(next_c)
        return next_h,next_c#(next_c,next_h)

