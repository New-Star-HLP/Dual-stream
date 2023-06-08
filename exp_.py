'''
to explor the effect
feedback+offset+focus
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class lstm_conv1d(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,kernel_size,stride):# kernle_size_hidden,stride_hidden
        super(lstm_conv1d, self).__init__()
        self.in_dim=in_dim
        self.hidden_dim=hidden_dim
        self.out_dim=out_dim
        self.kernel_size1=kernel_size
        self.stride1=stride
        self.out_conv1d=nn.Conv1d(in_channels=self.in_dim,out_channels=self.out_dim*4,
                           kernel_size=self.kernel_size1,stride=self.stride1)
    def forward(self,x,cur=None): #x: b n c/b c n,cur:b n c/b c n
        cur_c=cur  #假设只需要上时刻的状态, the dim is identated with the x
        if cur_c is None:
            cur_c=torch.zeros_like(x)
        #assert cur_h.size(-1)==self.hidden_dim, 'the hidden_dim is not matching'
        x1=self.out_conv1d(x)
        c_i,c_f,c_o,c_g=torch.splite(x1,4,dim=1)
        i=torch.sigmoid(c_i)
        f=torch.sigmoid(c_f)
        o=torch.sigmoid(c_o)
        g=torch.tanh(c_g)
        next_c=f*cur_c+i*g
        next_h=o*torch.tanh(next_c)
        return next_h,next_c

class Q_vector(nn.Module): #the Q_vector can share
    def __init__(self,in_dim,out_dim,):
        super(Q_vector, self).__init__()
        self.Q_vector=lstm_conv1d(in_dim=in_dim,hidden_dim=in_dim,
                                  out_dim=out_dim,kernel_size=1,stride=1)#nn.Linear(dim,dim)
        # nn.Conv1d(dim,dim,1,1) / Mlp_1d(dim,4*dim,dim
    def forward(self,x,c_cur=None):
        return self.Q_vector(x,c_cur)

class Patm(nn.Module):
    def __init__(self,dim,qkv_bias=False,mode="fc",proj_drop=0.,):
        super(Patm, self).__init__()
        self.fc_h = nn.Conv1d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_c = nn.Conv1d(dim, dim, 1, 1, bias=qkv_bias)

        self.tfc_h = nn.Conv1d(2 * dim, dim, 7, stride=1, padding=(0, 7 // 2), groups=dim, bias=False)

        self.reweight = nn.Sequential(nn.Conv1d(dim,dim//4,1,1),nn.Conv1d(dim//4,3*dim,1,1))
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode

        if mode == 'fc':
            self.theta_h_conv = nn.Sequential(nn.Conv1d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv1d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
    def forward(self, x):
        b,c,n= x.shape
        theta_h = self.theta_h_conv(x) # phase: theta

        x_h = self.fc_h(x)  # height amplitude
        x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)

        h = self.tfc_h(x_h)  # token height
        c = self.fc_c(x)  # b c n
        a = F.adaptive_avg_pool1d(h + c, output_size=2)
        return torch.mean(a,dim=1).reshape(-1,2) #b 2

class Deformable_(nn.Module):
    def __init__(self,dim):
        super(Deformable_, self).__init__()
        self.offset=Patm(dim)
        self.lin=nn.Linear(100,2)
    @torch.no_grad()
    def reference_points(self,b,device,dtype):
        x,y=torch.meshgrid(
            torch.linspace(-0.99,0.4,224,device=device,dtype=dtype),
            torch.linspace(-0.99,0.4,224,device=device,dtype=dtype)
        )
        ref=torch.stack((x,y),dim=-1)
        return ref[None,...].expand(b,-1,-1,-1)
    def forward(self,q_vector,class_feedback=None):
        b,c,n=q_vector.size()
        device,dtype=q_vector.device,q_vector.dtype
        offset=torch.relu(self.offset(q_vector))
        if class_feedback is None:
            class_feedback=torch.zeros((b,100)).to(device)
        offset=offset+torch.relu(self.lin(class_feedback))
        offset=offset.mul_(0.1).clamp_(0.,0.6).reshape(b,1,1,2)
        ref=self.reference_points(b,device,dtype)
        pos=ref+offset
        return pos
class Focus_deformable(nn.Module):
    def __init__(self,dim):
        super(Focus_deformable, self).__init__()
        self.offset_focus=nn.Sequential(nn.Conv2d(dim,6,5,3),nn.AdaptiveMaxPool2d([28,28]))
        self.reset_parameters()
    def reset_parameters(self):
        for m in self.parameters():
            if isinstance(m,(nn.Linear,nn.Conv2d)):
                nn.init.constant_(m.weights,0.)
                nn.init.constant_(m.bias,0.)
    @torch.no_grad()
    def reference(self,B,device,dtype):#b 196 351
        x,y=torch.meshgrid(
            torch.linspace(-0.95,-0.7,27,device,dtype),
            torch.linspace(-0.95,-0.7,13,device,dtype)
        )
        ref=torch.stack((x,y),dim=-1)
        return ref[None,...].expand(B,-1,-1,-1)
    def forward(self,x):
        B,C,H,W=x.size()
        device,dtype=x.device,x.dtype
        offset=torch.relu(self.offset_focus(x)).transpose(3,1)#b 27 13 6-->b 27 13 2
        offset=torch.cat((torch.mean(offset[...,:3],dim=-1,keepdim=True),
                         torch.mean(offset[...,3:],dim=-1,keepdim=True)),dim=-1)
        ref=self.reference(B,device,dtype)
        pos=ref+offset
        '''x_=F.grid_sample(q_vector.reshape(-1,1,-1,-1),pos[...,(-1,0)],'linear',
                         align_corners=True)#.reshape(B,20,16)
        x_=rearrange(x_,'b c n h->b (c h) n')'''
        return pos#x_

class Patch_Embedding(nn.Module):
    def __init__(self,dim,patch=(32,16),stride=(16,8)):
        super(Patch_Embedding, self).__init__()
        self.proj_=nn.Conv2d(3,dim,patch,stride)
        self.norm=nn.LayerNorm(dim)
        self.proj_out=nn.Linear(dim,dim)
    def forward(self,imgs,**kwargs):
        x=self.proj_(imgs).flatten(2)#.transpose(1,2)
        if hasattr(self,"norm"):
            x=self.norm(self.proj_out(x))
        return x
class Attn(nn.Module):#
    def __init__(self,dim):
        super(Attn, self).__init__()
        self.q=Q_vector(dim)
        self.k_vector=nn.Linear(dim,dim)
        self.v_vector=nn.Linear(dim,dim)
        self.scale=dim**-0.5
    def forward(self,x): #x:b 351,196
        q_v=self.q(x)
        k_v=self.k_vector(x)
        v_v=self.v_vector(x)
        attn=torch.matmul(torch.matmul(q_v,k_v.transpose(-1,-2))*self.scale,v_v)
        return attn
class Block(nn.Module):#dim:list len(dim)==depth
    def __init__(self,dim,depth,):
        super(Block, self).__init__()
        self.cls_token=nn.Parameter(torch.randn())
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.pos_emb=nn.Parameter(torch.linspace(0.1,19.6,196,device=device))##
        assert len(dim)==depth, "the len of dim(list) is not matching with the depth"
        self.transformer=nn.Sequential(*[nn.LayerNorm(dim[i]),Attn(dim[i]),nn.Linear(dim[i],4*dim[i]),
                                      nn.Linear(4*dim[i],dim[i]),nn.LayerNorm(dim)] for i in range(depth))
    def forward(self,x):
        return self.transformer(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.patch=Patch_Embedding(196)
        self.q=Q_vector(in_dim=196,out_dim=196)
        self.off=Deformable_(dim=196)
        self.off_1=Focus_deformable(dim=196)
        self.proj_=nn.Sequential(nn.Conv2d)
        device="cuda" if torch.cuda.is_available() else "cpu"
        self.cls_token=nn.Parameter(torch.randn(1,1,352)).to(device)
        self.block=Block(dim=[196,196,196],depth=3)
        self.fc=nn.Linear(353,200)
    def forward(self,imgs):
        x=self.patch(imgs)  #b 196 741
        class_feedback=None
        c_cur=None
        c_cur_=None
        for i in range(3):
            q,c_cur=self.q(x,c_cur)
            pos_=self.off(q,class_feedback)
            x_=F.grid_sample(
                input=imgs, grid=pos_[..., (1, 0)], mode='bilinear', align_corners=True)
            x1=self.patch(x_)  #b 196 351
            q_,c_cur_=self.q(x1,c_cur_) #
            pos_1=self.off_1(x1)
            x1_ = F.grid_sample(  #b 3 27 13
                input=x_, grid=pos_1[..., (1, 0)], mode='bilinear', align_corners=True)
            x1_1=self.patch(x1_) #b 196 1
            x2=torch.cat((x1,x1_1),dim=-1) #b 196 352
            x2=torch.cat((self.cls_token,x2),dim=-1)
            x3=self.block(x2)
            class_feedback=self.fc(x3[:,0])
            x4=class_feedback
        return x4







