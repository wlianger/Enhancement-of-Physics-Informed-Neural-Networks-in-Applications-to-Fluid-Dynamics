import argparse
import csv
from datetime import datetime
import time
import numpy as np
import torch
from torch.optim import lr_scheduler

from MyDataset.generate_data import boundary_data, equation_data, supervision_data
from network_structure.MLP import PINN_net

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
def parse_args():
    parser = argparse.ArgumentParser(description='Training Flow Using GB-PINN.')
    parser.add_argument('--num_epochs', type=int, default=70001, help='Number of epochs to train ')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of num_samples ')
    parser.add_argument('--alpha', type=float, default=0.2, help='损失函数计算过程当中的平衡超参数')
    parser.add_argument('--learning_rate', type=float, default=0.5 * 1e-3, help='Learning rate ')
    parser.add_argument('--Dimension_U', type=float, default=0.025,
                        help='Dimensionless processing parameters ')
    parser.add_argument('--Dimension_L', type=float, default=0.05,
                        help='Dimensionless processing parameters')
    parser.add_argument('--miu', type=float, default=1e-6, help='运动粘度：1e-6')
    parser.add_argument('--data_path', type=str, default='../../data/dataset/u=0.5/',
                        help='Path to the data ')
    parser.add_argument('--loss_path', type=str, default='../../data/loss/u=0.5/0.0510/',
                        help='Path to the data ')
    parser.add_argument('--model_path', type=str, default='../../model/u=0.5/0.0510/',
                        help='Path to save the trained model')

    return parser.parse_args()


class PINN_Train(torch.nn.Module):
    def __init__(self, model):
        super(PINN_Train, self).__init__()
        # assign the architectures
        self.model = model
        # assign the weights for each task
        self.weights = torch.nn.Parameter(torch.ones(model.output_size-1).float())
        self.mse_cost_function = torch.nn.MSELoss(reduction='mean')


    def forward(self, args):
        mse_uz, mse_us, mse_ur = self.calculate_loss(args.num_samples, domain, input_data, real_data, args.Dimension_U, args.Dimension_L, args.miu)
        task_losses = [mse_uz, mse_us, mse_ur]
        task_losses = torch.stack(task_losses)

        return task_losses

    def get_last_shared_layer(self):
        return self.model.get_penultimate_layer()

    def pde(self, input, u, l, miu):
        U = u
        L = l
        miu = miu
        x = input

        result = self.model(x)

        uz = result[:, 0].unsqueeze(-1)
        us = result[:, 1].unsqueeze(-1)
        ur = result[:, 2].unsqueeze(-1)
        p = result[:, 3].unsqueeze(-1)

        z = x[:, 0].unsqueeze(-1)
        r = x[:, 1].unsqueeze(-1)

        a1 = r * ur
        a2 = r * uz

        ur_rz = torch.autograd.grad(ur, x, grad_outputs=torch.ones_like(ur),
                                    create_graph=True, allow_unused=True)[0]
        ur_z = ur_rz[:, 0].unsqueeze(-1)
        ur_r = ur_rz[:, 1].unsqueeze(-1)
        ur_zz = torch.autograd.grad(ur_z, x, grad_outputs=torch.ones_like(ur_z),
                                    create_graph=True, allow_unused=True)[0][:, 0].unsqueeze(-1)
        ur_rr = torch.autograd.grad(ur_r, x, grad_outputs=torch.ones_like(ur_r),
                                    create_graph=True, allow_unused=True)[0][:, 1].unsqueeze(-1)

        us_rz = torch.autograd.grad(us, x, grad_outputs=torch.ones_like(us),
                                    create_graph=True, allow_unused=True)[0]
        us_z = us_rz[:, 0].unsqueeze(-1)
        us_r = us_rz[:, 1].unsqueeze(-1)
        us_zz = torch.autograd.grad(us_z, x, grad_outputs=torch.ones_like(us_z),
                                    create_graph=True, allow_unused=True)[0][:, 0].unsqueeze(-1)
        us_rr = torch.autograd.grad(us_r, x, grad_outputs=torch.ones_like(us_r),
                                    create_graph=True, allow_unused=True)[0][:, 1].unsqueeze(-1)

        uz_rz = torch.autograd.grad(uz, x, grad_outputs=torch.ones_like(uz),
                                    create_graph=True, allow_unused=True)[0]
        uz_z = uz_rz[:, 0].unsqueeze(-1)
        uz_r = uz_rz[:, 1].unsqueeze(-1)
        uz_zz = torch.autograd.grad(uz_z, x, grad_outputs=torch.ones_like(uz_z),
                                    create_graph=True, allow_unused=True)[0][:, 0].unsqueeze(-1)
        uz_rr = torch.autograd.grad(uz_r, x, grad_outputs=torch.ones_like(uz_r),
                                    create_graph=True, allow_unused=True)[0][:, 1].unsqueeze(-1)

        p_rz = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p),
                                   create_graph=True, allow_unused=True)[0]
        p_z = p_rz[:, 0].unsqueeze(-1)
        p_r = p_rz[:, 1].unsqueeze(-1)

        da1 = torch.autograd.grad(a1, x, grad_outputs=torch.ones_like(a1), create_graph=True, allow_unused=True)[0][:,
              1].unsqueeze(-1)
        da2 = torch.autograd.grad(a2, x, grad_outputs=torch.ones_like(a2), create_graph=True, retain_graph=True)[0][:,
              0].unsqueeze(-1)

        eq1 = ur * ur_r + uz * ur_z - us * us / r + p_r - miu / (U * L * r) * ur_r - miu / (U * L) * ur_rr - miu / (
                    U * L) * ur_zz + miu / (U * L) * ur / (r * r)
        eq2 = ur * us_r + uz * us_z + ur * us / r - miu / (U * L * r) * us_r - miu / (U * L) * us_rr - miu / (
                    U * L) * us_zz + miu / (U * L) * us / (r * r)
        eq3 = ur * uz_r + uz * uz_z + p_z - miu / (U * L * r) * uz_r - miu / (U * L) * uz_rr - miu / (U * L) * uz_zz
        eq4 = da1 + da2
        eqs = torch.concat((eq1, eq2, eq3, eq4), dim=1)
        return eqs

    def calculate_loss(self, num_samples, domain, input_data, real_data, u, l, miu):
        velocity_value = torch.ones((500, 1)).to(device)
        velocity_zeros_100 = torch.zeros((100, 1)).to(device)
        velocity_zeros_500 = torch.zeros((500, 1)).to(device)
        eq_zeros = torch.zeros((5000, 4)).to(device)

        bc1, bc2, bc3, bc4, bc5 = boundary_data(num_samples, domain)
        f1 = equation_data(num_samples, domain)

        net_bc_1 = self.model(bc1)  # z=0.00; r(0.05，0.056)
        mse_uz_1 = self.mse_cost_function(net_bc_1[:, 0].unsqueeze(-1), velocity_zeros_100)
        mse_us_1 = self.mse_cost_function(net_bc_1[:, 1].unsqueeze(-1), velocity_zeros_100)
        mse_ur_1 = self.mse_cost_function(net_bc_1[:, 2].unsqueeze(-1), velocity_zeros_100)

        net_bc_2 = self.model(bc2)  # z=0.12; r(0.05，0.056)
        mse_uz_2 = self.mse_cost_function(net_bc_2[:, 0].unsqueeze(-1), velocity_zeros_100)
        mse_us_2 = self.mse_cost_function(net_bc_2[:, 1].unsqueeze(-1), velocity_zeros_100)
        mse_ur_2 = self.mse_cost_function(net_bc_2[:, 2].unsqueeze(-1), velocity_zeros_100)

        net_bc_3 = self.model(bc3)  # z(0.0,0.12); r=0.05
        mse_uz_3 = self.mse_cost_function(net_bc_3[:, 0].unsqueeze(-1), velocity_zeros_500)
        mse_us_3 = self.mse_cost_function(net_bc_3[:, 1].unsqueeze(-1), velocity_value)
        mse_ur_3 = self.mse_cost_function(net_bc_3[:, 2].unsqueeze(-1), velocity_zeros_500)

        net_bc_4 = self.model(bc4)  # z(0.0,0.12); r=0.056
        mse_uz_4 = self.mse_cost_function(net_bc_4[:, 0].unsqueeze(-1), velocity_zeros_500)
        mse_us_4 = self.mse_cost_function(net_bc_4[:, 1].unsqueeze(-1), velocity_zeros_500)
        mse_ur_4 = self.mse_cost_function(net_bc_4[:, 2].unsqueeze(-1), velocity_zeros_500)

        net_bc_5 = self.model(bc5)  # z=0; r=0.05
        mse_p_5 = self.mse_cost_function(net_bc_5[:, 3].unsqueeze(-1), velocity_zeros_500)

        net_in_6 = self.model(input_data)
        # mse_data_uz = self.mse_cost_function(net_in_6[:, 0], real_data[:, 0])
        # mse_data_us = self.mse_cost_function(net_in_6[:, 1], real_data[:, 1])
        # mse_data_ur = self.mse_cost_function(net_in_6[:, 2], real_data[:, 2])
        # mse_data_p = self.mse_cost_function(net_in_6[:, 3], real_data[:, 3])
        mse_data = self.mse_cost_function(net_in_6, real_data)

        f_out1 = self.pde(f1, u, l, miu)
        # mse_f_uz = self.mse_cost_function(f_out1[:, 0], eq_zeros[:, 0])
        # mse_f_us = self.mse_cost_function(f_out1[:, 1], eq_zeros[:, 1])
        # mse_f_ur = self.mse_cost_function(f_out1[:, 2], eq_ze ros[:, 2])
        # mse_f_p = self.mse_cost_function(f_out1[:, 3], eq_zeros[:, 3])
        mse_f = self.mse_cost_function(f_out1, eq_zeros)
        #
        # return mse_uz_1 + mse_uz_2 + mse_uz_3 + mse_uz_4 + mse_data_uz + mse_f_uz , \
        #        mse_us_1 + mse_us_2 + mse_us_3 + mse_us_4 + mse_data_us + mse_f_us , \
        #        mse_ur_1 + mse_ur_2 + mse_ur_3 + mse_ur_4 + mse_data_ur + mse_f_ur ,\
        #        mse_f_p + mse_data_p + mse_p_5
        return mse_uz_1+mse_ur_1+mse_us_1+mse_uz_2+mse_ur_2+mse_us_2+mse_uz_3+mse_ur_3+mse_us_3+mse_uz_4+mse_ur_4+mse_us_4+mse_p_5,\
               mse_data, mse_f



args = parse_args()
net = PINN_net(2, 256, 4, 7)
model = PINN_Train(net).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

wz = ws = wr = wp = 0
initial_loss = torch.empty(1, 3)
task_losses = torch.ones(1, 3)

domain = torch.tensor([0.0, 0.12/args.Dimension_L, 0.05/args.Dimension_L, 0.056/args.Dimension_L])

T_num = (int)(args.num_epochs/3)
cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_num)
pre_loss = float('inf')

torch.cuda.synchronize()
time_start = time.time()

input_data, real_data = supervision_data(args.data_path+"TC_train_0.0510.csv", args.Dimension_U, args.Dimension_L)
v_input_data, v_real_data = supervision_data(args.data_path+"TC_verify.csv", args.Dimension_U, args.Dimension_L)

current_time = datetime.now()
filename = current_time.strftime("%Y-%m-%d")

headers = ['iteration', 'total', 'bc', 'data', 'f', 'error_l2']
with open(args.loss_path+filename+"-0.0510-loss-grad.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)

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


    #加入gradnorm


    optimizer.step()
    cosine_scheduler.step()


    #归一化处理


    wz = model.weights.data.cpu().numpy()[0]
    ws = model.weights.data.cpu().numpy()[1]
    wr = model.weights.data.cpu().numpy()[2]

    with torch.autograd.no_grad():
            net_val = net(v_input_data)
            val_loss = torch.norm(net_val - v_real_data, p='fro')
            val_reloss = val_loss / torch.norm(v_real_data, p='fro')
            print(epoch, "Training Loss:", loss.data.item(), " ; Verifying Loss:", val_loss.data.item(), " ; Verifying reLoss:", val_reloss.data.item())
            print("   weight_bc:", wz, " ; weight_data:", ws, " ; weight_f:", wr)
            print("   loss:",  task_losses[0].item(), " ; loss_data:",  task_losses[1].item(), " ; loss_f:",  task_losses[2].item())
            if (val_loss) < pre_loss:
                pre_loss = val_loss
                torch.save(net.state_dict(), args.model_path+filename+"-0.0510-grad_model")
            if (epoch+1) % 100 == 0:
                number_loss = [epoch, loss.data.item(), task_losses[0].item(),  task_losses[1].item(),  task_losses[2].item(), val_reloss.data.item()]
                with open(args.loss_path+filename+"-0.0510-loss-grad.csv", 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(number_loss)
torch.cuda.synchronize()
time_end = time.time()
time_sum = time_end - time_start
number = time_sum/60/60
file_name = "0.0510-time-consuming-grad.txt"
with open(args.loss_path+file_name, "w") as f:
    f.write(filename+"time-consuming: "+str(number)+" hours")


