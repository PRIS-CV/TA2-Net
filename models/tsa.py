'''
tsa.py
Created by Wei-Hong Li [https://weihonglee.github.io]
This code allows you to attach task-specific parameters, including adapters, pre-classifier alignment (PA) mapping
from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
(https://arxiv.org/pdf/2103.13841.pdf), to a pretrained backbone. 
It only learns attached task-specific parameters from scratch on the support set to adapt 
the pretrained model for previously unseen task with very few labeled samples.
'Cross-domain Few-shot Learning with Task-specific Adapters.' (https://arxiv.org/pdf/2107.00358.pdf)
'''

from torch.autograd import Variable
from pickletools import read_int4
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math
from config import args
import copy
import torch.nn.functional as F
from models.losses import prototype_loss, compute_prototypes
from utils import device
from gumbelmodule import GumbleSoftmax







class conv_env_max(nn.Module):
    def __init__(self, orig_conv, activation_rates):
        super(conv_env_max, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        self.activation_rates  = activation_rates
        self.drop = nn.Dropout(p=0)
        self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
        self.alpha.requires_grad = True
        self.garm = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
        self.garm.requires_grad = True
        self.rarm = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
        self.rarm.requires_grad = True

    def forward(self, x):
        y = self.conv(x)

        y1 = F.conv2d(x, self.alpha, stride=self.conv.stride)
        y2 = F.conv2d(x, self.garm, stride=self.conv.stride) 
        y3 = F.conv2d(x, self.rarm, stride=self.conv.stride) 

        if self.activation_rates ==0:
            y = y 
        if self.activation_rates==1:
            y = y +y1
        if self.activation_rates==2:
            y = y + 1/2 * (y2+y1)
        if self.activation_rates==3:
            y = y + 1/3 * (y3+y2+y1)
        return y
    
class conv_eval(nn.Module):
    def __init__(self, orig_conv, activation_rates):
        super(conv_eval, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        self.activation_rates  = activation_rates
        self.drop = nn.Dropout(p=0)
        self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
        self.alpha.requires_grad = True
        self.garm = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
        self.garm.requires_grad = True
        self.rarm = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
        self.rarm.requires_grad = True

    def forward(self, x):
        y = self.conv(x)

        y1 = F.conv2d(x, self.alpha, stride=self.conv.stride)
        y2 = F.conv2d(x, self.garm, stride=self.conv.stride) 
        y3 = F.conv2d(x, self.rarm, stride=self.conv.stride) 

        if self.activation_rates ==0:
            y = y 
        if self.activation_rates==1:
            y = y +y1
        if self.activation_rates==2:
            y = y + 1/2 * (y2+y1)
        if self.activation_rates==3:
            y = y + 1/3 * (y3+y2+y1)
        return y


class conv_agent(nn.Module):
    def __init__(self, orig_conv):
        super(conv_agent, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride

        self.relu = nn.ReLU(inplace=True)
        self.conv_fc =  nn.Conv2d(in_planes, 16, kernel_size=1)
        self.bn_fc = nn.BatchNorm2d(16)
        self.conv_fc_1 =  nn.Conv2d(16, 4, kernel_size=1)


        self.alpha = nn.Conv2d(in_planes, planes, kernel_size=1,  stride= stride)
        self.alpha.requires_grad = True
        self.garm = nn.Conv2d(in_planes, planes, kernel_size=1, stride= stride)
        self.garm.requires_grad = True
        self.rarm = nn.Conv2d(in_planes, planes, kernel_size=1,  stride= stride)
        self.rarm.requires_grad = True
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 1:
                nn.init.xavier_normal_(m.weight)


    def forward(self, x, context_labels, n_way):
        y = self.conv(x)

        hard_w = F.avg_pool2d(x, x.size(2)).flatten(1)
        hard_w = compute_prototypes(hard_w, context_labels, n_way)
        hard_w = hard_w.unsqueeze(-1).unsqueeze(-1)
        hard_w = self.relu(self.bn_fc (self.conv_fc(hard_w)))
        hard_w =  (self.conv_fc_1(hard_w))
        hard_w = torch.mean(hard_w, dim=0).view(-1).unsqueeze(0)
        hard_w = F.softmax(hard_w/10)

        m = torch.distributions.Categorical((hard_w))
        action = m.sample()
        y1 = self.alpha(x)
        y2 = self.garm(x)
        y3 = self.rarm(x)
        
        loss = m.log_prob(action).unsqueeze(-1)

        if hard_w[:,0].unsqueeze(0)>= torch.max(hard_w):

            max_action = hard_w[:,0].unsqueeze(0).data.clone().zero_()
            
        if hard_w[:,1].unsqueeze(0)>= torch.max(hard_w):

            max_action =  hard_w[:,1].unsqueeze(0).data.clone().zero_()+1
            y = y + y1
        if hard_w[:,2].unsqueeze(0)>= torch.max(hard_w):

            max_action = hard_w[:,2].unsqueeze(0).data.clone().zero_()+2
            y = y + 1/2 * (y2+y1)
        if hard_w[:,3].unsqueeze(0)>= torch.max(hard_w):

            max_action = hard_w[:,3].unsqueeze(0).data.clone().zero_()+3
            y = y + 1/3 * (y3+y2+y1)

        return y,  max_action, action, loss
        
        
        


class conv_env_sam(nn.Module):
    def __init__(self, orig_conv, activation_rates):
        super(conv_env_sam, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        self.activation_rates  = activation_rates

            
        self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
        self.alpha.requires_grad = True
        self.garm = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
        self.garm.requires_grad = True
        self.rarm = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
        self.rarm.requires_grad = True

    def forward(self, x):
        y = self.conv(x)

        y1 = F.conv2d(x, self.alpha, stride=self.conv.stride)
        y2 = F.conv2d(x, self.garm, stride=self.conv.stride) 
        y3 = F.conv2d(x, self.rarm, stride=self.conv.stride) 


        
        if self.activation_rates ==0:
            y = y 
        if self.activation_rates ==1:
            y = y + y1
        if self.activation_rates ==2:
            y = y + 1/2 *( y2+y1)
        if self.activation_rates ==3:
            y = y +  1/3 * ( y3+y2+y1)

        return y







class resnet_agent(nn.Module):
    """ Attaching task-specific adapters (alpha) and/or PA (beta) to the ResNet backbone """
    def __init__(self, orig_resnet):
        super(resnet_agent, self).__init__()
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
                v.requires_grad= False
        
        for block in orig_resnet.layer1:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_agent(m)
                    setattr(block, name, new_conv)
        for block in orig_resnet.layer2:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_agent(m)
                    setattr(block, name, new_conv)
        for block in orig_resnet.layer3:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_agent(m)
                    setattr(block, name, new_conv)
        for block in orig_resnet.layer4:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_agent(m)
                    setattr(block, name, new_conv)
        
        self.backbone = orig_resnet
        self.relu = nn.ReLU(inplace=True)



    def forward(self, x):
        return self.backbone.forward(x=x)

    def embed(self, x, context_labels, n_way):
        
        p = [] 

        for module_pos, module in self.backbone._modules.items():
            if module_pos != "layer4" and module_pos != "layer3" and module_pos != "layer2" and module_pos != "layer1" :
                x = module(x)
            else:
                
                for module_pos1, module1 in module._modules.items():
                #    x = module(x)
            #        from IPython import embed;embed();
                    if module_pos1=="0":
                        for module_pos2, module2 in module1._modules.items():
                            if module_pos2 == "conv1":
                                identity = x
                                x, A, action, loss= module2(x, context_labels, n_way)
                                p.append(action)
                                if  module_pos == "layer1":
                                    loss_all = loss

                                    gate_activations = A
                                else:
                                    loss_all  = torch.cat((loss_all, loss),-1)
                                    gate_activations = torch.cat((gate_activations, A),1)
                            elif   module_pos2 == "conv2":
                                x, A, action, loss= module2(x,context_labels, n_way)
                                p.append(action)
                                loss_all  = torch.cat((loss_all, loss),-1)
                                gate_activations = torch.cat((gate_activations, A),1)
                            elif   module_pos2 == "downsample":
                                identity = module2(identity)
                            else:
                                x = module2(x)
                        x = self.relu(x + identity)

                    else :
                        
                        for module_pos2, module2 in module1._modules.items():
                   
                            if module_pos2 == "conv1":
                                identity = x
                                x, A, action, loss= module2(x,context_labels, n_way)
                                p.append(action)
                                loss_all  = torch.cat((loss_all, loss),-1)
                                gate_activations = torch.cat((gate_activations, A),1)

                            elif   module_pos2 == "conv2":
                                x,A, action, loss= module2(x, context_labels, n_way)
                                p.append(action)
                                gate_activations = torch.cat((gate_activations, A),1)
                                loss_all  = torch.cat((loss_all, loss),-1)
                            else:
                                x = module2(x)
                        x = self.relu(x + identity)
                
        return x, gate_activations, p, loss_all

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.backbone.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.backbone.named_parameters()]

    def reset(self):      

        for k, v in self.backbone.named_parameters():
            if 'bn_fc' in k:
                nn.init.constant_(v.data[0], 1)
                nn.init.constant_(v.data[1], 0)

        


class resnet_env_sam(nn.Module):
    """ Attaching task-specific adapters (alpha) and/or PA (beta) to the ResNet backbone """
    def __init__(self, orig_resnet, activation_rates):
        super(resnet_env_sam, self).__init__()
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
                v.requires_grad= False
        iter = 0 

        for block in orig_resnet.layer1:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:

                    new_conv = conv_env_sam(m, activation_rates[iter])
                    setattr(block, name, new_conv)
                    iter += 1

        for block in orig_resnet.layer2:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:

                    new_conv = conv_env_sam(m, activation_rates[iter])
                    setattr(block, name, new_conv)
                    iter += 1
        for block in orig_resnet.layer3:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:

                    new_conv = conv_env_sam(m, activation_rates[iter])
                    setattr(block, name, new_conv)
                    iter += 1


        for block in orig_resnet.layer4:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:

                    new_conv = conv_env_sam(m, activation_rates[iter])
                    setattr(block, name, new_conv)
                    iter += 1
        
        self.backbone = orig_resnet
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.backbone.forward(x=x)

    def embed(self, x):
        
        return self.backbone.embed(x)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.backbone.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.backbone.named_parameters()]

    def reset(self):
        # initialize task-specific adapters (alpha)
        for k, v in self.backbone.named_parameters():
            if 'alpha' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001
            if 'garm' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001
            if 'rarm' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001
            if 'barm' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001

class resnet_env_max(nn.Module):
    """ Attaching task-specific adapters (alpha) and/or PA (beta) to the ResNet backbone """
    def __init__(self, orig_resnet, activation_rates):
        super(resnet_env_max, self).__init__()
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
                v.requires_grad= False
        iter = 0 
        for block in orig_resnet.layer1:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_env_max(m, activation_rates[:, iter])
                    setattr(block, name, new_conv)
                    iter += 1
        for block in orig_resnet.layer2:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:

                    new_conv = conv_env_max(m, activation_rates[:, iter])
                    setattr(block, name, new_conv)
                    iter += 1
        for block in orig_resnet.layer3:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_env_max(m, activation_rates[:, iter])
                    setattr(block, name, new_conv)
                    iter += 1


        for block in orig_resnet.layer4:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:

                    new_conv = conv_env_max(m, activation_rates[:, iter])
                    setattr(block, name, new_conv)
                    iter += 1
        
        self.backbone = orig_resnet

    def forward(self, x):
        return self.backbone.forward(x=x)

    def embed(self, x):
        return self.backbone.embed(x)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.backbone.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.backbone.named_parameters()]

    def reset(self):
        # initialize task-specific adapters (alpha)
        for k, v in self.backbone.named_parameters():
            if 'alpha' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001
            if 'garm' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001
            if 'rarm' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001
            
class pa(nn.Module):
    """ 
    pre-classifier alignment (PA) mapping from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
    (https://arxiv.org/pdf/2103.13841.pdf)
    """
    def __init__(self, feat_dim):
        super(pa, self).__init__()
        # define pre-classifier alignment mapping
        self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim, 1, 1))
        self.weight.requires_grad = True

    def forward(self, x):
        if len(list(x.size())) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.conv2d(x, self.weight.to(x.device)).flatten(1)
        return x
    
class resnet_eval(nn.Module):
    """ Attaching task-specific adapters (alpha) and/or PA (beta) to the ResNet backbone """
    def __init__(self, orig_resnet, activation_rates):
        super(resnet_eval, self).__init__()
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
                v.requires_grad= False
        iter = 0 
        for block in orig_resnet.layer1:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_eval(m, activation_rates[:, iter])
                    setattr(block, name, new_conv)
                    iter += 1
        for block in orig_resnet.layer2:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_eval(m, activation_rates[:, iter])
                    setattr(block, name, new_conv)
                    iter += 1
        for block in orig_resnet.layer3:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_eval(m, activation_rates[:, iter])
                    setattr(block, name, new_conv)
                    iter += 1


        for block in orig_resnet.layer4:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_eval(m, activation_rates[:, iter])
                    setattr(block, name, new_conv)
                    iter += 1
        self.backbone = orig_resnet

        feat_dim = orig_resnet.layer4[-1].bn2.num_features
        beta = pa(feat_dim)
        setattr(self, 'beta', beta)
   

    def forward(self, x):
        return self.backbone.forward(x=x)

    def embed(self, x):
        return self.backbone.embed(x)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.backbone.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.backbone.named_parameters()]

    def reset(self):
        # initialize task-specific adapters (alpha)
        for k, v in self.backbone.named_parameters():
            if 'alpha' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001
            if 'garm' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001
            if 'rarm' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001

        v = self.beta.weight
        self.beta.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)


def env_train(context_images, context_labels, model, max_iter=40, lr=0.1,  distance='cos'):
    """
    Optimizing task-specific parameters attached to the ResNet backbone, 
    e.g. adapters (alpha) and/or pre-classifier alignment mapping (beta)
    """
    model.eval()
    alpha_params = [v for k, v in model.named_parameters() if 'alpha' in k]
    garm_params = [v for k, v in model.named_parameters() if 'garm' in k]
    rarm_params = [v for k, v in model.named_parameters() if 'rarm' in k]

    params = []
    params.append({'params': alpha_params})
    params.append({'params': garm_params})
    params.append({'params': rarm_params})




    optimizer = torch.optim.Adadelta(params, lr=lr) 


    for i in range(max_iter):
        optimizer.zero_grad()
        model.zero_grad()
        aligned_features = model.embed(context_images)
        loss, stat, _ = prototype_loss(aligned_features, context_labels,
                                       aligned_features, context_labels, distance=distance)
        loss.backward()
        optimizer.step()

    return

def val_te_train(context_images, context_labels, model, max_iter=40, lr=0.1, lr_beta=1, distance='cos'):
    """
    Optimizing task-specific parameters attached to the ResNet backbone, 
    e.g. adapters (alpha) and/or pre-classifier alignment mapping (beta)
    """
    model.eval()
    alpha_params = [v for k, v in model.named_parameters() if 'alpha' in k]
    garm_params = [v for k, v in model.named_parameters() if 'garm' in k]
    rarm_params = [v for k, v in model.named_parameters() if 'rarm' in k]
    beta_params = [v for k, v in model.named_parameters() if 'beta' in k]
    params = []
    params.append({'params': alpha_params})
    params.append({'params': garm_params})
    params.append({'params': rarm_params})
    params.append({'params': beta_params, 'lr': lr_beta})



    optimizer = torch.optim.Adadelta(params, lr=lr) 


    for i in range(max_iter):
        optimizer.zero_grad()
        model.zero_grad()
        context_features = model.embed(context_images)
        aligned_features = model.beta(context_features)
        
        loss, stat, _ = prototype_loss(aligned_features, context_labels,
                                       aligned_features, context_labels, distance=distance)
        loss.backward()
        optimizer.step()

    return


