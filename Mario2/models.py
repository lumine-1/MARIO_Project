import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class OCTResNet101(nn.Module):
    def __init__(self):
        super(OCTResNet101, self).__init__()
        self.model = models.resnet101(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改第一层以接受单通道输入
        self.model.layer4[2].add_module("cbam", CBAM(2048))  # 在最后一层添加CBAM
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)  # 修改最后一层全连接层以适应3分类任务

    def forward(self, image):
        x = self.model(image)
        return x


###################上面是一种


# class OCTResNet101_n(nn.Module):
#     def __init__(self):
#         super(OCTResNet101_n, self).__init__()
#         self.model = models.resnet101(pretrained=True)
#         self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改第一层以接受单通道输入
#         self.model.fc = nn.Linear(self.model.fc.in_features, 3)  # 修改最后一层全连接层以适应3分类任务
#
#     def forward(self, image):
#         x = self.model(image)
#         return x


class MultiHeadAttention_resnet(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super(MultiHeadAttention_resnet, self).__init__()
        assert in_dim % num_heads == 0, "in_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, in_dim = x.size()
        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attention = self.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5))
        attention_output = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, seq_len, in_dim)

        return attention_output


class OCTResNet101_n(nn.Module):
    def __init__(self):
        super(OCTResNet101_n, self).__init__()
        self.model = models.resnet101(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改第一层以接受单通道输入
        # 添加一个Multihead Attention层
        self.attention = MultiHeadAttention_resnet(in_dim=1024, num_heads=8)
        # 修改最后一层全连接层以适应3分类任务
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)

    def forward(self, image):
        # 通过ResNet前几层
        x = self.model.conv1(image)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)

        # 将ResNet的输出转化为Multihead Attention的输入
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2,
                                     1)  # (batch_size, channels, height*width) -> (batch_size, height*width, channels)
        # Multihead Attention
        x = self.attention(x)
        # 还原为原始形状
        x = x.permute(0, 2, 1).view(b, c, h,
                                    w)  # (batch_size, height*width, channels) -> (batch_size, channels, height, width)
        # 通过ResNet的剩余层
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return x


class ConvNeXtLarge_attention(nn.Module):
    def __init__(self, num_classes=3, num_heads=8, embed_dim=1536):  # 调整embed_dim为1536
        super(ConvNeXtLarge_attention, self).__init__()
        self.convnext = models.convnext_large(pretrained=True)
        self.convnext.classifier = nn.Identity()  # 移除原始分类头
        # 多头注意力机制
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # 将单通道图像转换为三通道图像
        x = x.repeat(1, 3, 1, 1)
        x = self.convnext(x)

        # 假设输出的形状为 (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape
        # 转换维度以适应多头注意力机制 (batch_size, seq_len, embed_dim)
        x = x.view(batch_size, channels, -1).permute(2, 0, 1)  # (seq_len, batch_size, embed_dim)
        attn_output, _ = self.multihead_attention(x, x, x)
        # 恢复维度 (batch_size, seq_len, embed_dim)
        attn_output = attn_output.permute(1, 0, 2).contiguous()
        # 平均池化和分类
        x = torch.mean(attn_output, dim=1)
        x = self.classifier(x)
        return x


###################下面是用于整体训练的

class OCTResNet50(nn.Module):
    def __init__(self):
        super(OCTResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Identity()  # 移除最后的全连接层以提取特征

    def forward(self, images):
        batch_size, num_images, channels, height, width = images.shape
        images = images.view(-1, channels, height, width)  # 将批次展平为单个大批次
        features = self.model(images)  # 获取特征
        features = features.view(batch_size, num_images, -1)  # 恢复为原来的批次结构
        features = torch.mean(features, dim=1)  # 对所有图像的特征取平均
        return features  # 返回特征而不是直接通过全连接层


class OCTResNet50C(nn.Module):
    def __init__(self):
        super(OCTResNet50C, self).__init__()
        self.base_model = OCTResNet50()
        self.fc = nn.Linear(2048, 3)  # 2048 是 ResNet50 提取的特征数量

    def forward(self, images):
        features = self.base_model(images)
        output = self.fc(features)  # 使用平均特征进行分类
        return output


class OCTConvNeXt(nn.Module):
    def __init__(self):
        super(OCTConvNeXt, self).__init__()
        self.model = models.convnext_base(pretrained=True)
        # 修改第一层以接受单通道输入
        self.model.features[0][0] = nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
        # 移除最后的分类器，以便于特征提取
        self.model.classifier = nn.Identity()

    def forward(self, images):
        batch_size, num_images, channels, height, width = images.shape
        images = images.view(-1, channels, height, width)  # 将批次展平为单个大批次
        features = self.model(images)  # 获取特征
        features = features.view(batch_size, num_images, -1)  # 恢复为原来的批次结构
        features = torch.mean(features, dim=1)  # 对所有图像的特征取平均
        return features  # 返回特征而不是直接通过全连接层


class OCTConvNeXtC(nn.Module):
    def __init__(self):
        super(OCTConvNeXtC, self).__init__()
        self.base_model = OCTConvNeXt()
        # ConvNeXt base 最终的特征数量是1024，修改最后一层以适应3分类任务
        self.fc = nn.Linear(1024, 3)

    def forward(self, images):
        features = self.base_model(images)
        output = self.fc(features)  # 使用平均特征进行分类
        return output





