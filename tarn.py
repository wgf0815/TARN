import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from feature_extractor import Conv3x3Block
from collections import OrderedDict



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size


    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape
            
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size-1) / 2)
        right_padding = int(self.block_size / 2)
        
        batch_size, channels, height, width = mask.shape
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1), # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size), #- left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size**2, 2).cuda().long(), offsets.long()), 1)
        
        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            
        block_mask = 1 - padded_mask#[:height, :width]
        return block_mask
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False,
                 block_size=1,max_pool=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.max_pool = max_pool

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        
        if self.max_pool:
            out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

class ResNet(nn.Module):

    def __init__(self, block, n_blocks, drop_rate=0.0, dropblock_size=5, max_pool=True):
        super(ResNet, self).__init__()

        self.inplanes = 3
        self.layer1 = self._make_layer(block, n_blocks[0], 64,
                                       stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 160,
                                       stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], 320,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, n_blocks[3], 640,
                                       stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size,max_pool=max_pool)

        self.drop_rate = drop_rate
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1,max_pool=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size,max_pool=max_pool)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class resRN(nn.Module):
    def __init__(self, block, n_blocks, drop_rate=0.0, dropblock_size=5, max_pool=True):
        super(resRN,self).__init__()
        self.inplanes=640*2
        self.layer1 = self._make_layer(block, n_blocks[0], 640,
                                       stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 640,
                                       stride=2, drop_rate=drop_rate)
        self.fc_input_c = 640
        _half = 40
        self.fc = nn.Sequential(nn.Linear(self.fc_input_c,_half),
                                nn.ReLU(),
                                nn.Linear(_half, 1))
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1,max_pool=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size,max_pool=max_pool)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        b, c, h, w = x2.size()
        s, _, _, _ = x1.size()
        query_s = x2.unsqueeze(1).repeat(1, s, 1, 1, 1).contiguous().view(b * s, c , h, w)
        support_q = x1.unsqueeze(0).repeat(b, 1, 1, 1, 1).contiguous().view(b * s, c , h, w)
        out1 = torch.cat([query_s, support_q], dim=1)
        out2 = torch.cat([support_q, query_s], dim=1)

        out = self.layer1(out1)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out).view(b,s)
        out1 = self.logsoftmax(out) 
        
        out = self.layer1(out2)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out).view(b,s)
        out2 = self.logsoftmax(out)

        return [out1,out2]

class resTCA(nn.Module):
    def __init__(self,feature_dim, block, n_blocks, drop_rate=0.0, dropblock_size=5, max_pool=True):
        super(resTCA, self).__init__()
        self.in_channels = feature_dim
        self.squeeze = resTCAModule(feature_dim, block, n_blocks, drop_rate=0.0, dropblock_size=5, max_pool=True)

        self.inplanes=feature_dim*4
        self.layer1 = self._make_layer(block, n_blocks[0], 640,
                                       stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 640,
                                       stride=2, drop_rate=drop_rate)
        self.fc_input_c = 640
        _half = 40
        self.fc = nn.Sequential(nn.Linear(self.fc_input_c,_half),
                                nn.ReLU(),
                                nn.Linear(_half, 1))
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1,max_pool=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size,max_pool=max_pool)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        b, c, h, w = x2.size()
        s, _, _, _ = x1.size()
        support_q,query_s,Ass,Asq = self.squeeze(x1,x2) #[b*s,c*2,h,w]
        out1 = torch.cat([query_s, support_q], dim=1)
        out2 = torch.cat([support_q, query_s], dim=1)

        out = self.layer1(out1)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out).view(b,s)
        out1 = self.logsoftmax(out) 
        
        out = self.layer1(out2)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out).view(b,s)
        out2 = self.logsoftmax(out)

        Ass = torch.log(Ass)
        Asq = torch.log(Asq)
        Asq = Asq.permute(1,0)

        return [out1, out2,Ass,Asq]

class resTCAModule(nn.Module):
    def __init__(self, feature_dim, block, n_blocks, drop_rate=0.0, dropblock_size=5, max_pool=True):
        super(resTCAModule,self).__init__()
        self.feature_dim = feature_dim
        self.prerelation = resPreRelation(feature_dim, block, n_blocks, drop_rate=0.0, dropblock_size=5, max_pool=True)
        self.theta = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(self.feature_dim, momentum=0.1, affine=True),
        )
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(self.feature_dim, momentum=0.1, affine=True),
        )
    def forward(self,support_data,query_data):
        b, c, h, w = query_data.size()
        s, _, _, _ = support_data.size()

        Ass,Asq = self.prerelation(support_data,query_data)  #[s,b]
        # A = torch.cat([torch.diag_embed(torch.diag(Ass)),Asq],dim=1) #[s,s+b]
        # A = F.normalize(A,dim=1,p=2)
        with torch.no_grad():
            A = torch.cat([torch.diag_embed(torch.diag(Ass)),Asq],dim=1) #[s,s+b]
            A = F.normalize(A,dim=1,p=2)
        global_data = torch.cat([support_data,query_data],dim=0).contiguous().view(s+b,-1)
        proto = torch.matmul(A,global_data).contiguous().view(s,c,h,w)

        theta_proto = self.theta(proto).view(s,c,-1)
        theta_proto = F.normalize(theta_proto,dim=1)
        theta_query = self.theta(query_data).view(b,c,-1)
        theta_support = self.theta(support_data).view(s,c,-1)
        theta_query = F.normalize(theta_query,dim=1)
        theta_support = F.normalize(theta_support,dim=1)
        phi_support = theta_support.permute(0,2,1)
        phi_query = theta_query.permute(0,2,1)
        support_cross = torch.matmul(phi_support,theta_proto) #[s,hw,hw]
        query_cross = torch.matmul(phi_query.unsqueeze(1).repeat(1,s,1,1),
                                    theta_proto.unsqueeze(0).repeat(b,1,1,1)) #[b,s,hw,hw]
        global_proto = proto.permute(0,2,3,1).contiguous().view(s,-1,c)
        support_cross_ = torch.matmul(support_cross,global_proto).permute(0,2,1).contiguous().view(s,c,h,w)
        query_cross_ = torch.matmul(query_cross,global_proto.unsqueeze(0)).permute(0,1,3,2).contiguous().view(b*s,c,h,w)
        W_support = self.W(support_cross_)
        W_query = self.W(query_cross_)

        new_support = torch.cat([support_data,W_support],dim=1) 
        new_support = new_support.unsqueeze(0).repeat(b, 1, 1, 1, 1).contiguous().view(b * s, c*2, h, w)
        new_query = torch.cat([query_data.unsqueeze(1).repeat(1,s,1,1,1).contiguous().view(b*s,c,h,w),
                                W_query],dim=1)

        return new_support,new_query,Ass,Asq

class resPreRelation(nn.Module):
    def __init__(self, feature_dim, block, n_blocks, drop_rate=0.0, dropblock_size=5, max_pool=True):
        super(resPreRelation, self).__init__()
        self.in_channels = feature_dim

        self.inplanes=feature_dim*2
        self.layer1 = self._make_layer(block, n_blocks[0], 640,
                                       stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 640,
                                       stride=2, drop_rate=drop_rate)
        self.fc_input_c = 640
        _half = 40
        self.fc = nn.Sequential(nn.Linear(self.fc_input_c,_half),
                                nn.ReLU(),
                                nn.Linear(_half, 1))
        self.softmax = nn.Softmax(dim=0)
        
    
    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1,max_pool=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size,max_pool=max_pool)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate)
        layers.append(layer)
        self.inplanes = planes * block.expansion

        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        b, c, h, w = x2.size()
        s, _, _, _ = x1.size()

        ss = torch.cat([x1.unsqueeze(1).repeat(1,s,1,1,1),
                        x1.unsqueeze(0).repeat(s,1,1,1,1)],dim=2).view(-1,c*2,h,w) #[s*s,c*2,h,w]
        sq = torch.cat([x1.unsqueeze(1).repeat(1,b,1,1,1),
                        x2.unsqueeze(0).repeat(s,1,1,1,1)],dim=2).view(-1,c*2,h,w) #[s*b,c*2,h,w]
        outss = self.layer2(self.layer1(ss)).view(s*s,-1)
        outss = self.fc(outss).view(s,s)

        outsq = self.layer2(self.layer1(sq)).view(s*b,-1)
        outsq = self.fc(outsq).view(s,b)

        outss = self.softmax(outss)
        outsq = self.softmax(outsq)

        return outss,outsq

class RN(nn.Module):
    def __init__(self, feature_dim):
        super(RN, self).__init__()
        self.in_channels = feature_dim
       
        self.layer1 = Conv3x3Block(self.in_channels*2, feature_dim, is_pool=True)
        self.layer2 = Conv3x3Block(feature_dim, feature_dim, is_pool=True)
        self.fc = nn.Sequential(nn.Linear(feature_dim*3*3, 8),
                                nn.ReLU(),
                                nn.Linear(8, 1),
                                nn.Sigmoid())

    def forward(self, x1, x2): 

        b, c, h, w = x2.size()
        s, _, _, _ = x1.size()
        query_s = x2.unsqueeze(1).repeat(1, s, 1, 1, 1).contiguous().view(b * s, c , h, w)
        support_q = x1.unsqueeze(0).repeat(b, 1, 1, 1, 1).contiguous().view(b * s, c , h, w)
        out1 = torch.cat([query_s, support_q], dim=1)
        out2 = torch.cat([support_q, query_s], dim=1)

        out=self.layer1(out1)
        out=self.layer2(out)
        out = out.view(out.size(0),-1)
        out1 = self.fc(out).view(b,s)

        out = self.layer1(out2)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out2 = self.fc(out).view(b,s)

        return [out1,out2]

class TCA(nn.Module):
    def __init__(self, feature_dim):
        super(TCA, self).__init__()
        self.in_channels = feature_dim

        self.layer1 = Conv3x3Block(self.in_channels*4, feature_dim, is_pool=True)
        self.layer2 = Conv3x3Block(feature_dim, feature_dim, is_pool=True)
        self.fc = nn.Sequential(nn.Linear(feature_dim*3*3, 8),
                                nn.ReLU(),
                                nn.Linear(8, 1),
                                nn.Sigmoid())
        self.squeeze = TCAModule(self.in_channels)

    def forward(self, x1, x2):
        b, c, h, w = x2.size()
        s, _, _, _ = x1.size()
        support_q,query_s,Ass,Asq = self.squeeze(x1,x2) #[b*s,c*2,h,w]
        out1 = torch.cat([query_s, support_q], dim=1)
        out2 = torch.cat([support_q, query_s], dim=1)

        out = self.layer1(out1)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out1 = self.fc(out)
        out1 = out1.view(b,s)
        
        out = self.layer1(out2)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out2 = self.fc(out)
        out2 = out2.view(b,s)

        return [out1, out2,Ass,Asq]

class TCAModule(nn.Module):
    def __init__(self,feature_dim):
        super(TCAModule,self).__init__()
        self.feature_dim = feature_dim
        self.prerelation = PreRelation(feature_dim)
        self.theta = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(self.feature_dim, momentum=0.1, affine=True),
        )
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(self.feature_dim, momentum=0.1, affine=True),
        )
    def forward(self,support_data,query_data):
        b, c, h, w = query_data.size()
        s, _, _, _ = support_data.size()

        Ass,Asq = self.prerelation(support_data,query_data)  #[s,b]
        # A = torch.cat([torch.diag_embed(torch.diag(Ass)),Asq],dim=1) #[s,s+b]
        # A = F.normalize(A,dim=1,p=1)
        with torch.no_grad():
            A = torch.cat([torch.diag_embed(torch.diag(Ass)),Asq],dim=1) #[s,s+b]
            A = F.normalize(A,dim=1,p=1)
        global_data = torch.cat([support_data,query_data],dim=0).contiguous().view(s+b,-1)
        proto = torch.matmul(A,global_data).contiguous().view(s,c,h,w)

        theta_proto = self.theta(proto).view(s,c,-1)
        theta_proto = F.normalize(theta_proto,dim=1)
        theta_query = self.theta(query_data).view(b,c,-1)
        theta_support = self.theta(support_data).view(s,c,-1)
        theta_query = F.normalize(theta_query,dim=1)
        theta_support = F.normalize(theta_support,dim=1)
        phi_support = theta_support.permute(0,2,1)
        phi_query = theta_query.permute(0,2,1)
        support_cross = torch.matmul(phi_support,theta_proto) #[s,hw,hw]
        query_cross = torch.matmul(phi_query.unsqueeze(1).repeat(1,s,1,1),
                                    theta_proto.unsqueeze(0).repeat(b,1,1,1)) #[b,s,hw,hw]
        global_proto = proto.permute(0,2,3,1).contiguous().view(s,-1,c)
        support_cross_ = torch.matmul(support_cross,global_proto).permute(0,2,1).contiguous().view(s,c,h,w)
        query_cross_ = torch.matmul(query_cross,global_proto.unsqueeze(0)).permute(0,1,3,2).contiguous().view(b*s,c,h,w)
        W_support = self.W(support_cross_)
        W_query = self.W(query_cross_)

        new_support = torch.cat([support_data,W_support],dim=1) 
        new_support = new_support.unsqueeze(0).repeat(b, 1, 1, 1, 1).contiguous().view(b * s, c*2, h, w)
        new_query = torch.cat([query_data.unsqueeze(1).repeat(1,s,1,1,1).contiguous().view(b*s,c,h,w),
                                W_query],dim=1)
        
        return new_support,new_query,Ass,Asq

class PreRelation(nn.Module):
    def __init__(self, feature_dim):
        super(PreRelation, self).__init__()
        self.in_channels = feature_dim

        self.layer1 = Conv3x3Block(self.in_channels * 2, feature_dim, is_pool=True)
        self.layer2 = Conv3x3Block(feature_dim, feature_dim, is_pool=True)
        self.fc = nn.Sequential(nn.Linear(feature_dim * 3 * 3, 8),
                                nn.ReLU(),
                                nn.Linear(8, 1),
                                nn.Sigmoid())
    def forward(self, x1, x2):
        b, c, h, w = x2.size()
        s, _, _, _ = x1.size()

        ss = torch.cat([x1.unsqueeze(1).repeat(1,s,1,1,1),
                        x1.unsqueeze(0).repeat(s,1,1,1,1)],dim=2).view(-1,c*2,h,w) #[s*s,c*2,h,w]
        sq = torch.cat([x1.unsqueeze(1).repeat(1,b,1,1,1),
                        x2.unsqueeze(0).repeat(s,1,1,1,1)],dim=2).view(-1,c*2,h,w) #[s*b,c*2,h,w]

        outss = self.layer1(ss)
        outss = self.layer2(outss).view(s*s,-1)
        outss = self.fc(outss).view(s,s)

        outsq = self.layer1(sq)
        outsq = self.layer2(outsq).view(s*b,-1)
        outsq = self.fc(outsq).view(s,b)

        outss = F.normalize(outss,dim=1,p=1)
        outsq = F.normalize(outsq,dim=0,p=1)

        return outss,outsq

if __name__ == '__main__':
    state_dict = torch.load('frn_pretrain/mini-ImageNet/model.pth')
    new_state = OrderedDict()
    for k,v in state_dict.items():
        if k[:18] == 'feature_extractor.':
            new_state[k[18:]]=v
    model = ResNet(BasicBlock, [1, 1, 1, 1], drop_rate=0.0, max_pool=True)
    model.load_state_dict(new_state,strict=True)
    data = torch.randn(2, 3, 84, 84)
    x = model(data)
    print(x.shape)
    resrn = resRN(BasicBlock,[1,1],drop_rate=0.0,max_pool=True)
    y1=resrn(x,x)
    print(y1[0].shape)