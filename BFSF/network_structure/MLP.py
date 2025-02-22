from torch import nn

class PINN_net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(PINN_net, self).__init__()
        # 定义一个空的Sequential模型
        self.model = nn.ModuleList()
        self.output_size = output_size
        # 添加输入层
        self.model.append(nn.Linear(input_size, hidden_size))
        self.model.append(nn.Tanh())


        # 循环添加隐藏层和激活函数
        for _ in range(num_layers - 1):
            self.model.append(nn.Linear(hidden_size, hidden_size))
            self.model.append(nn.Tanh())

        # 添加输出层
        self.model.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

    def get_penultimate_layer(self):
        return self.model[-1]


