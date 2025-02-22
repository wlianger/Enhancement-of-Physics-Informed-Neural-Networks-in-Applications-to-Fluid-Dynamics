import argparse
import csv
from datetime import datetime
import time
import numpy as np
import torch
from torch.optim import lr_scheduler
from MyDataset.generate_data import boundary_data, equation_data, supervision_data
from network_structure.MLP_Xavier import PINN_net

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
def parse_args():
    parser = argparse.ArgumentParser(description='Training Backward-facing step Flow Using PINN.')
    parser.add_argument('--num_epochs', type=int, default=40001, help='Number of epochs to train (default: 20001)')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of num_samples (default: 5000)')
    parser.add_argument('--alpha', type=float, default=0.3, help='损失函数计算过程当中的平衡超参数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate (default: 0.001)')
    parser.add_argument('--Dimension_U', type=float, default=0.02, help='Dimensionless processing parameters (default: 0.02)')
    parser.add_argument('--Dimension_L', type=float, default=0.01, help='Dimensionless processing parameters (default: 0.01)')
    parser.add_argument('--weight_eq', type=float, default=1, help='weight_eq (default: 1)')
    parser.add_argument('--weight_bc', type=float, default=1, help='weight_bc (default: 1)')
    parser.add_argument('--weight_data', type=float, default=1, help='weight_data (default: 1)')
    parser.add_argument('--Reynolds_number', type=float, default=200, help='Re (default: 497.61)')
    parser.add_argument('--data_path', type=str, default='../data/dataset/U=0.02-decrease/', help='Path to the data ')
    parser.add_argument('--loss_path', type=str, default='../data/loss/U=0.02-decrease/ninit/', help='Path to the data ')
    parser.add_argument('--model_path', type=str, default='../model/u=0.02-decrease/ninit/', help='Path to save the trained model')

    return parser.parse_args()

class PINN_Train(torch.nn.Module):
    def __init__(self, model):
        super(PINN_Train, self).__init__()
        self.model = model
        self.weights = torch.ones(model.output_size).float().to(device)
        self.mse_cost_function = torch.nn.MSELoss(reduction='mean')


    def forward(self, args):
        mse_ux, mse_uy, mse_p = self.calculate_loss(args.num_samples, domain, input_data, real_data, args.Reynolds_number)
        task_losses = [mse_ux, mse_uy, mse_p]
        task_losses = torch.stack(task_losses)

        return task_losses

    def get_last_shared_layer(self):
        return self.model.get_penultimate_layer()

    def pde(self, input, Re):
        res = self.model(input)

        u = res[:, 0].unsqueeze(-1)
        v = res[:, 1].unsqueeze(-1)
        p = res[:, 2].unsqueeze(-1)

        u_xy = torch.autograd.grad(u, input, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = u_xy[:, 0].unsqueeze(-1)
        u_y = u_xy[:, 1].unsqueeze(-1)
        u_xx = torch.autograd.grad(u_x, input, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0].unsqueeze(
            -1)
        u_yy = torch.autograd.grad(u_y, input, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1].unsqueeze(
            -1)

        v_xy = torch.autograd.grad(v, input, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = v_xy[:, 0].unsqueeze(-1)
        v_y = v_xy[:, 1].unsqueeze(-1)
        v_xx = torch.autograd.grad(v_x, input, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0].unsqueeze(
            -1)
        v_yy = torch.autograd.grad(v_y, input, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1].unsqueeze(
            -1)

        p_xy = torch.autograd.grad(p, input, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_x = p_xy[:, 0].unsqueeze(-1)
        p_y = p_xy[:, 1].unsqueeze(-1)

        eq1 = u * u_x + v * u_y + p_x - 1 / Re * (u_xx + u_yy)
        eq2 = u * v_x + v * v_y + p_y - 1 / Re * (v_xx + v_yy)
        eq3 = u_x + v_y

        eqs = torch.concat((eq1, eq2, eq3), dim=1)
        return eqs

    def calculate_loss(self, num_samples, domain, input_data, real_data, Re):

        velocity_value = torch.ones((50, 1)).to(device)
        velocity_zeros = torch.zeros((50, 1)).to(device)
        velocity_zeros_1 = torch.zeros((500, 1)).to(device)
        velocity_zeros_2 = torch.zeros((334, 1)).to(device)
        velocity_zeros_3 = torch.zeros((166, 1)).to(device)
        velocity_zeros_4 = torch.zeros((100, 1)).to(device)

        bc1, bc2, bc3, bc4, bc5, bc6 = boundary_data(num_samples, domain)
        f1, f2 = equation_data(num_samples, domain)

        net_bc_1 = self.model(bc1)  # x:0 y:[0,1]
        mse_ux_1 = self.mse_cost_function(net_bc_1[:, 0].unsqueeze(-1), velocity_value)
        mse_uy_1 = self.mse_cost_function(net_bc_1[:, 1].unsqueeze(-1), velocity_zeros)

        net_bc_2 = self.model(bc2)  # x:[0,26.7] y:1
        mse_uy_2 = self.mse_cost_function(net_bc_2[:, 1].unsqueeze(-1), velocity_zeros_1)
        mse_ux_2 = self.mse_cost_function(net_bc_2[:, 0].unsqueeze(-1), velocity_zeros_1)

        net_bc_3 = self.model(bc3)  # x:[8.9,26.7] y:-1
        mse_ux_3 = self.mse_cost_function(net_bc_3[:, 0].unsqueeze(-1), velocity_zeros_2)
        mse_uy_3 = self.mse_cost_function(net_bc_3[:, 1].unsqueeze(-1), velocity_zeros_2)

        net_bc_4 = self.model(bc4)  # x:[0.0,8.9] y:0
        mse_ux_4 = self.mse_cost_function(net_bc_4[:, 0].unsqueeze(-1), velocity_zeros_3)
        mse_uy_4 = self.mse_cost_function(net_bc_4[:, 1].unsqueeze(-1), velocity_zeros_3)

        net_bc_5 = self.model(bc5)  # x=8.9 y[-1,0]
        mse_ux_5 = self.mse_cost_function(net_bc_5[:, 0].unsqueeze(-1), velocity_zeros)
        mse_uy_5 = self.mse_cost_function(net_bc_5[:, 1].unsqueeze(-1), velocity_zeros)

        net_bc_p = self.model(bc6)  # x=0.0; y=0.0
        mse_p = self.mse_cost_function(net_bc_p[:, 2].unsqueeze(-1), velocity_zeros_4)

        net_in_6 = self.model(input_data)
        mse_data = self.mse_cost_function(net_in_6, real_data)

        f_out1 = self.pde(f1, Re)
        f_out2 = self.pde(f2, Re)
        num1 = f_out1.shape[0]
        eq_zeros = torch.zeros((num1, 3)).to(device)
        mse_f_1 = self.mse_cost_function(f_out1, eq_zeros)
        num2 = f_out2.shape[0]
        eq_zeros = torch.zeros((num2, 3)).to(device)
        mse_f_2 = self.mse_cost_function(f_out2, eq_zeros)

        return mse_ux_1 + mse_uy_1 + mse_ux_2 + mse_uy_2 + mse_ux_3 + mse_uy_3 + mse_uy_4 + mse_ux_4 + mse_ux_5 + mse_uy_5 + mse_p, mse_data, mse_f_2 + mse_f_1

args = parse_args()
net =PINN_net(2, 128, 3, 7)
model = PINN_Train(net).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

wx = wy = wp = 0
initial_loss = torch.empty(1, 3)
task_losses = torch.ones(1, 3)

domain = torch.tensor([-8.9, 0.0, 17.8, -1.0, 1.0])

T_num = (int)(args.num_epochs)
cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_num)
pre_loss = float('inf')

torch.cuda.synchronize()
time_start = time.time()

input_data, real_data = supervision_data(args.data_path+"BF_train_0.0085.csv", args.Dimension_U, args.Dimension_L)
v_input_data, v_real_data = supervision_data(args.data_path+"BF_verify.csv", args.Dimension_U, args.Dimension_L)

current_time = datetime.now()

filename = current_time.strftime("%Y-%m-%d")

headers = ['iteration', 'total', 'bc', 'data', 'f', 'error_l2']
with open(args.loss_path+filename+"-loss-0.0085.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)

model_num = 0

for epoch in range(args.num_epochs):
    optimizer.zero_grad()
    task_losses = model(args)
    weighted_loss = torch.mul(model.weights, task_losses)

    if epoch == 0:
        if torch.cuda.is_available():
            initial_loss = task_losses.data.cpu()
        else:
            initial_loss = task_losses.data

        initial_loss = initial_loss.numpy()

    loss = torch.sum(weighted_loss)


    loss.backward(retain_graph=True)
    optimizer.step()
    cosine_scheduler.step()

    wx = model.weights.data.cpu().numpy()[0]
    wy = model.weights.data.cpu().numpy()[1]
    wp = model.weights.data.cpu().numpy()[2]

    with torch.autograd.no_grad():
        net_val = net(v_input_data)
        val_err = torch.norm(net_val.flatten() - v_real_data.flatten(), p='fro')
        val_loss = val_err/torch.norm(v_real_data.flatten(), p='fro')
        print(epoch, "Training Loss:", loss.data.item(), " ; Verifying Loss:", val_loss.data.item(), " ; Verifying ab_err:", val_err.data.item())
        print("   weight_bc:", wx, " ; weight_data:", wy, " ; weight_f:", wp)
        if (val_loss) < pre_loss or model_num>=100:
            pre_loss = val_err
            torch.save(net, args.model_path+filename+"-0.0085_model.pth")
            model_num = 0
        if (epoch+1) % 100 == 0:
            number_loss = [epoch, loss.data.item(),  task_losses[0].item(), task_losses[1].item(), task_losses[2].item(), val_loss.data.item()]
            with open(args.loss_path+filename+"-loss-0.0085.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(number_loss)

torch.cuda.synchronize()
time_end = time.time()
time_sum = time_end - time_start
number = time_sum/60/60
file_name = "time-consuming-0.0085.txt"
with open(args.loss_path+file_name, "w") as f:
    f.write(filename+"time-consuming-0.0085: "+str(number)+" hours")


