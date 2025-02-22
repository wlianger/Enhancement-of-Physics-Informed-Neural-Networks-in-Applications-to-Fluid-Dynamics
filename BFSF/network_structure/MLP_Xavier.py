import torch
import torch.nn as nn
import torch.nn.init as init


class PINN_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(PINN_net, self).__init__()
        # 确保输入维度与第一个线性层匹配
        assert input_size == 2, "Expected input dimension to be 2, but got {}".format(input_size)
        self.output_size = output_size
        # 使用Sequential构建模型
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Tanh())
        # 添加多个隐藏层
        for _ in range(5):  # 假设我们想要5个隐藏层
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        # 最后一个线性层连接到输出层

        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        # 检查输入维度
        if x.size(1) != 2:
            raise ValueError("Expected input dimension to be 2, got {}".format(x.size(1)))
        out = self.model(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.0)

    def get_penultimate_layer(self):
        return self.model[-1]

