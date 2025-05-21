import torch
import torch.nn as nn
import torch.optim as optim
from modules import DeepONet, init_he

model = DeepONet(
    layer_sizes_branch=[100,64,64,100],
    layer_sizes_trunk=[1,64,64,100],
    activation="sigmoid",
    kernel_initializer=init_he
)


mse = nn.MSELoss()
def customLoss(y_pred, y_true, x):
    mse_loss = mse(y_pred, y_true)
    dy = torch.autograd.grad(y_pred, x, grad_outputs=torch.ones_like(y_pred),
                        create_graph=True, retain_graph=True)[0]
    dy2 = torch.autograd.grad(dy, x, grad_outputs=torch.ones_like(dy),
                        create_graph=True, retain_graph=True)[0]
    residual = dy2 + y_pred
    return mse(residual, torch.zeros_like(residual)) + mse_loss
    #return mse_loss

optimizer = optim.Adam(model.parameters(), lr = 0.001)


def true_function(x):
    return torch.sin(x)

x_func = torch.sin(torch.linspace(0,2*torch.pi,100)).unsqueeze(0).repeat(32,1)
x_loc = torch.rand(32, 1, requires_grad=True) * 2 * torch.pi

y_true = true_function(x_loc)

model.train()
for epoch in range(10000):
    optimizer.zero_grad()

    y_pred = model((x_func, x_loc))
    loss = customLoss(y_pred, y_true, x_loc)
    
    loss.backward(retain_graph=True)
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

model.eval()
with torch.no_grad():
    test_loc = torch.linspace(0, 2 * torch.pi, 100).unsqueeze(1)  # [100, 1]
    test_func = x_func[0].unsqueeze(0).repeat(100, 1)             # [100, 100]
    pred = model((test_func, test_loc))                           # [100, 1]

import matplotlib.pyplot as plt
plt.plot(test_loc.squeeze(), torch.sin(test_loc).squeeze(), label="True")
plt.plot(test_loc.squeeze(), pred.squeeze(), label="Predicted")
plt.legend()
plt.show()


#6585