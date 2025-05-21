import torch
import torch.nn as nn
import torch.optim as optim
from modules import DeepONet, init_he
import matplotlib.pyplot as plt

batch_size = 32
n_points = 100

model = DeepONet(
    layer_sizes_branch=[n_points, 128, 128, n_points], 
    layer_sizes_trunk=[1, 128, 128, n_points],
    activation="tanh",
    kernel_initializer=init_he
)

def sample_functions(batch_size, n_points):
    x = torch.linspace(0, 1, n_points).unsqueeze(0).repeat(batch_size, 1)
    freq1 = torch.rand(batch_size, 1) * 3 + 1
    freq2 = torch.rand(batch_size, 1) * 3 + 1
    ph1 = torch.rand(batch_size, 1) * 2 * torch.pi
    ph2 = torch.rand(batch_size, 1) * 2 * torch.pi
    f = torch.sin(2*torch.pi * freq1* x + ph1) + torch.sin(2*torch.pi * freq2* x + ph2)
    return x,f


optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',         # loss가 최소화되도록
    factor=0.7,         # lr을 0.5배로 줄임
    patience=500,       # 200 epoch 동안 개선 없으면 줄임
    verbose=True        # 줄어들 때마다 print 해줌
)

loss_fn = nn.MSELoss()

for epoch in range(30000):
    optimizer.zero_grad()
    
    x_f, f_sample = sample_functions(batch_size=batch_size, n_points=n_points)
    x_loc = x_f.unsqueeze(-1)                          # [B, N, 1]
    y_true = (f_sample ** 2).unsqueeze(-1)             # [B, N, 1]

    B, N = f_sample.shape

    branch_input = f_sample.unsqueeze(1).repeat(1, N, 1).reshape(B * N, -1)  # [B*N, N]
    trunk_input = x_loc.reshape(B * N, 1)                                    # [B*N, 1]
    y_true_flat = y_true.reshape(B * N, 1)                                   # [B*N, 1]

    y_pred = model((branch_input, trunk_input))
    loss = loss_fn(y_pred, y_true_flat)

    loss.backward()
    optimizer.step()
    scheduler.step(loss)  # loss는 scalar여야 함

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    if epoch % 5000 == 0:
        model.eval()
        with torch.no_grad():
            x_f, f_sample = sample_functions(batch_size=1, n_points=n_points)  # 하나의 함수
            x_loc = x_f.unsqueeze(-1)                                          # [1, N, 1]
            y_true = (f_sample ** 2).unsqueeze(-1)                             # [1, N, 1]

            # DeepONet 입력 형태로 변환
            branch_input = f_sample.unsqueeze(1).repeat(1, n_points, 1).reshape(n_points, -1)  # [N, N]
            trunk_input = x_loc.reshape(n_points, 1)                                           # [N, 1]

            y_pred = model((branch_input, trunk_input))  # [N, 1]

            # 시각화
            plt.plot(x_f.squeeze(), y_true.squeeze(), label="True f(x)^2")
            plt.plot(x_f.squeeze(), y_pred.squeeze(), label="Predicted")
            plt.xlabel("x")
            plt.ylabel("f(x)^2")
            plt.title("Function-to-Function Mapping")
            plt.legend()
            plt.grid(True)
            plt.show()
            
model.eval()
with torch.no_grad():
    x_f, f_sample = sample_functions(batch_size=1, n_points=n_points)  # 하나의 함수
    x_loc = x_f.unsqueeze(-1)                                          # [1, N, 1]
    y_true = (f_sample ** 2).unsqueeze(-1)                             # [1, N, 1]

    # DeepONet 입력 형태로 변환
    branch_input = f_sample.unsqueeze(1).repeat(1, n_points, 1).reshape(n_points, -1)  # [N, N]
    trunk_input = x_loc.reshape(n_points, 1)                                           # [N, 1]

    y_pred = model((branch_input, trunk_input))  # [N, 1]

    # 시각화
    plt.plot(x_f.squeeze(), y_true.squeeze(), label="True f(x)^2")
    plt.plot(x_f.squeeze(), y_pred.squeeze(), label="Predicted")
    plt.xlabel("x")
    plt.ylabel("f(x)^2")
    plt.title("Function-to-Function Mapping")
    plt.legend()
    plt.grid(True)
    plt.show()