import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision import models
from torchvision.models import ConvNeXt_Large_Weights, convnext_large, vit_b_16
import torch.nn.functional as F


# class ChangeFormer(nn.Module):
#     def __init__(self, img_size=224, num_classes=4, dim=512, depth=6, heads=8, mlp_dim=2048):
#         super(ChangeFormer, self).__init__()
#
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, dim, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 添加池化层减少特征图尺寸
#             nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)  # 再次添加池化层减少特征图尺寸
#         )
#
#         # 假设输入图像大小为224x224，经过两次池化后的特征图大小应为56x56
#         self.flatten_dim = (img_size // 4) * (img_size // 4)
#         self.dim = dim
#
#         self.positional_encoding = nn.Parameter(torch.zeros(1, self.flatten_dim * 2, dim))
#
#         self.transformer = nn.Transformer(dim, heads, depth)
#
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, mlp_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(mlp_dim, num_classes)
#         )
#
#     def forward(self, x1, x2):
#         b, c, h, w = x1.size()
#
#         x1 = self.encoder(x1)  # (b, dim, 56, 56)
#         x2 = self.encoder(x2)  # (b, dim, 56, 56)
#
#         # 将特征图展平并转换形状
#         x1 = x1.view(b, self.dim, -1).permute(2, 0, 1)  # (seq_len, batch, dim)
#         x2 = x2.view(b, self.dim, -1).permute(2, 0, 1)  # (seq_len, batch, dim)
#
#         x = torch.cat((x1, x2), dim=0)  # 在序列维度上拼接
#
#         # 添加位置编码
#         x = x + self.positional_encoding[:x.size(0), :, :]
#
#         x = self.transformer(x)
#
#         x = x.mean(dim=0)  # 对序列维度进行平均
#
#         return self.mlp_head(x)





class ChangeFormer_sub9(nn.Module):
    def __init__(self, num_classes=4):
        super(ChangeFormer_sub9, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer for classification
        self.fc = nn.Linear(256, num_classes)

    def forward(self, image_ti, image_ti_1):
        x = torch.cat((image_ti, image_ti_1), dim=1)
        x = self.encoder(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x


class ChangeFormer_sub10_11(nn.Module):
    def __init__(self, num_classes=4, d_model=2048, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.0):
        super(ChangeFormer_sub10_11, self).__init__()

        # 使用预训练的 ResNet101 作为特征提取器
        self.encoder = models.resnet101(pretrained=True)
        self.encoder.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改第一个卷积层以适应双通道图像
        self.encoder.fc = nn.Identity()  # 去掉 ResNet101 的最后一层全连接层

        # Transformer 编码器
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_encoder_layers
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, image_ti, image_ti_1):
        # Concatenate input images along the channel dimension
        x = torch.cat((image_ti, image_ti_1), dim=1)
        x = self.encoder(x)
        # Ensure the tensor has the correct shape for the transformer encoder
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(2, 0, 1)
        # Pass through the transformer encoder
        x = self.transformer(x)
        # Global average pooling
        x = x.mean(dim=0)
        # Fully connected layer
        x = self.fc(x)

        return x


class ChangeFormer(nn.Module):
    def __init__(self, num_classes=4, d_model=1536, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.0):
        super(ChangeFormer, self).__init__()

        # 使用预训练的 convnext_large 模型
        self.convnext = models.convnext_large(pretrained=True)

        # 去掉 convnext 的全连接层
        self.convnext.classifier = nn.Identity()

        # Transformer 编码器
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_encoder_layers
        )

        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层进行分类
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, image_ti, image_ti_1):
        # 将输入图像沿通道维度拼接，并复制一个通道以创建三通道输入
        x = torch.cat((image_ti, image_ti_1), dim=1)  # 拼接成双通道
        x = torch.cat((x, x[:, :1, :, :]), dim=1)  # 复制一个通道，变成三通道

        # 提取特征
        x = self.convnext.features(x)

        # 调整形状以匹配 Transformer 编码器输入
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, -1).permute(2, 0, 1)  # [batch_size, channels, height*width] -> [height*width, batch_size, channels]

        # 通过 Transformer 编码器
        x = self.transformer(x)

        # 全局平均池化
        x = x.permute(1, 2, 0).contiguous().view(batch_size, channels, height, width)  # 恢复形状 [height*width, batch_size, channels] -> [batch_size, channels, height, width]
        x = self.global_avg_pool(x)
        x = x.view(batch_size, -1)  # [batch_size, channels, 1, 1] -> [batch_size, channels]

        # 全连接层
        x = self.fc(x)

        return x


class SiameseNetwork_sub12(nn.Module):
    def __init__(self):
        super(SiameseNetwork_sub12, self).__init__()
        self.cnn = models.resnet101(pretrained=True)
        # 修改输入层以适应单通道图像
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改最后一层全连接层
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_ftrs, 256)
        # 添加分类器层
        self.classifier = nn.Linear(256 * 2, 4)  # 2 because concatenating outputs from both branches

    def forward_one(self, x):
        return self.cnn(x)

    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        # 将两个输出拼接在一起
        combined_output = torch.cat((output1, output2), dim=1)
        # 分类器输出
        classification_output = self.classifier(combined_output)
        return classification_output


class SiameseNetwork_sub13_14(nn.Module):
    def __init__(self):
        super(SiameseNetwork_sub13_14, self).__init__()
        self.cnn = models.convnext_large(pretrained=True)

        # 修改最后一层全连接层
        num_ftrs = self.cnn.classifier[2].in_features
        self.cnn.classifier[2] = nn.Linear(num_ftrs, 256)

        # 添加分类器层
        self.classifier = nn.Linear(256 * 2, 4)  # 2 because concatenating outputs from both branches

    def forward_one(self, x):
        # 将单通道图像复制为三通道图像
        x = x.repeat(1, 3, 1, 1)
        return self.cnn(x)

    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)

        # 将两个输出拼接在一起
        combined_output = torch.cat((output1, output2), dim=1)

        # 分类器输出
        classification_output = self.classifier(combined_output)
        return classification_output






















class Attention(nn.Module):
    def __init__(self, in_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention = self.softmax(torch.bmm(query.unsqueeze(1), key.unsqueeze(2)))
        attention_output = torch.bmm(attention, value.unsqueeze(1)).squeeze(1)

        return attention_output


class SiameseNetwork_sub15_16(nn.Module):
    def __init__(self):
        super(SiameseNetwork_sub15_16, self).__init__()
        self.cnn = models.convnext_large(pretrained=True)

        # 修改最后一层全连接层
        num_ftrs = self.cnn.classifier[2].in_features
        self.cnn.classifier[2] = nn.Linear(num_ftrs, 256)

        # 添加注意力层
        self.attention = Attention(256)

        # 添加分类器层
        self.classifier = nn.Linear(256 * 2, 4)  # 2 because concatenating outputs from both branches

    def forward_one(self, x):
        # 将单通道图像复制为三通道图像
        x = x.repeat(1, 3, 1, 1)
        cnn_output = self.cnn(x)

        # 应用注意力机制
        attention_output = self.attention(cnn_output)

        return attention_output

    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)

        # 将两个输出拼接在一起
        combined_output = torch.cat((output1, output2), dim=1)

        # 分类器输出
        classification_output = self.classifier(combined_output)
        return classification_output


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
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

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = models.convnext_large(pretrained=True)
        self.cnn.classifier[2] = nn.Linear(self.cnn.classifier[2].in_features, 256)

        # 添加多头注意力层
        self.attention = MultiHeadAttention(256)

        # 添加分类器层
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward_one(self, x):
        # 将单通道图像复制为三通道图像
        x = x.repeat(1, 3, 1, 1)
        cnn_output = self.cnn(x)

        # 应用注意力机制
        attention_output = self.attention(cnn_output.unsqueeze(1)).squeeze(1)

        return attention_output

    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)

        # 将两个输出拼接在一起
        combined_output = torch.cat((output1, output2), dim=1)

        # 分类器输出
        classification_output = self.classifier(combined_output)
        return classification_output

