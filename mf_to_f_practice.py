import torch
import torch.nn as nn
import torch.optim as optim
from modules import DeepONet, init_he
import matplotlib.pyplot as plt

# CUDA 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 32
n_points = 200

# 모델을 CUDA로 이동
model = DeepONet(
    layer_sizes_branch=[2 * n_points, 512, 1024, 512, 256],
    layer_sizes_trunk=[1, 512, 1024, 512, 256],
    activation="silu",
    kernel_initializer=init_he,
).to(device)


def sample_functions(batch_size, n_points):
    x = torch.linspace(0, 1, n_points).unsqueeze(0).repeat(batch_size, 1).to(device)
    freq1 = torch.rand(batch_size, 1, device=device) * 3 + 1
    freq2 = torch.rand(batch_size, 1, device=device) * 3 + 1
    ph1 = torch.rand(batch_size, 1, device=device) * 2 * torch.pi
    ph2 = torch.rand(batch_size, 1, device=device) * 2 * torch.pi
    f = torch.sin(2 * torch.pi * freq1 * x + ph1) + torch.sin(
        2 * torch.pi * freq2 * x + ph2
    )
    _freq1 = torch.rand(batch_size, 1, device=device) * 3 + 1
    _freq2 = torch.rand(batch_size, 1, device=device) * 3 + 1
    _ph1 = torch.rand(batch_size, 1, device=device) * 2 * torch.pi
    _ph2 = torch.rand(batch_size, 1, device=device) * 2 * torch.pi
    _f = torch.sin(2 * torch.pi * _freq1 * x + _ph1) + torch.sin(
        2 * torch.pi * _freq2 * x + _ph2
    )

    f_g = f * _f
    return x, f, _f, f_g


# Optimizer 및 Loss 함수 설정
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",  # loss가 최소화되도록
    factor=0.7,  # lr을 0.7배로 줄임
    patience=500,  # 500 epoch 동안 개선 없으면 줄임
    verbose=True,  # 줄어들 때마다 print 해줌
)
loss_fn = nn.MSELoss().to(device)

for epoch in range(50000):
    optimizer.zero_grad()

    x, f, g, f_g = sample_functions(batch_size=batch_size, n_points=n_points)
    x_loc = x.unsqueeze(-1)  # [B, N, 1]
    y_true = f_g.unsqueeze(-1)  # [B, N, 1]

    B, N = f.shape

    brch = torch.cat([f, g], dim=1)
    branch_input = brch.unsqueeze(1).repeat(1, N, 1).reshape(B * N, -1)  # [B*N, N]
    trunk_input = x_loc.reshape(B * N, 1)  # [B*N, 1]
    y_true_flat = y_true.reshape(B * N, 1)  # [B*N, 1]

    y_pred = model((branch_input, trunk_input))
    loss = loss_fn(y_pred, y_true_flat)

    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())  # loss는 scalar여야 함

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    if epoch % 5000 == 0:
        model.eval()
        with torch.no_grad():
            x, f, g, f_g = sample_functions(batch_size=1, n_points=n_points)
            x_loc = x.unsqueeze(-1)
            y_true = f_g.unsqueeze(-1)

            brch = torch.cat([f, g], dim=1)
            branch_input = (
                brch.unsqueeze(1).repeat(1, n_points, 1).reshape(n_points, -1)
            )
            trunk_input = x_loc.reshape(n_points, 1)

            y_pred = model((branch_input, trunk_input))  # [N, 1]

            # 시각화
            plt.plot(x.cpu().squeeze(), f_g.cpu().squeeze(), label="True f(x)*g(x)")
            plt.plot(x.cpu().squeeze(), y_pred.cpu().squeeze(), label="Predicted")
            plt.xlabel("x")
            plt.ylabel("f(x)*g(x)")
            plt.title("Function-to-Function Mapping")
            plt.legend()
            plt.grid(True)
            plt.show()

model.eval()
with torch.no_grad():
    x, f, g, f_g = sample_functions(batch_size=batch_size, n_points=n_points)
    x_loc = x.unsqueeze(-1)
    y_true = f_g.unsqueeze(-1)

    brch = torch.cat([f, g], dim=1)
    branch_input = brch.unsqueeze(1).repeat(1, n_points, 1).reshape(n_points, -1)
    trunk_input = x_loc.reshape(n_points, 1)

    y_pred = model((branch_input, trunk_input))

    # 시각화
    plt.plot(x.cpu().squeeze(), f_g.cpu().squeeze(), label="True f(x)^2")
    plt.plot(x.cpu().squeeze(), y_pred.cpu().squeeze(), label="Predicted")
    plt.xlabel("x")
    plt.ylabel("f(x)*g(x)")
    plt.title("Function-to-Function Mapping")
    plt.legend()
    plt.grid(True)
    plt.show()
