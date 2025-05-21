import torch
import torch.nn as nn
import activations

def init_he(weight):
    nn.init.kaiming_normal_(weight, nonlinearity='relu')

class FNN(nn.Module):
    def __init__(self, layer_sizes, activation, initializer=init_he):
        super().__init__()
        layers = []
        
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]
            linear = nn.Linear(in_size, out_size) # layer 사이 결합
            
            if initializer:
                initializer(linear.weight)
                nn.init.zeros_(linear.bias)
                
            layers.append(linear)
            layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            
            if i < len(layer_sizes) - 2:
                layers.append(activation)
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Module):
    def __init__(self, 
                 layer_sizes_branch, 
                 layer_sizes_trunk, 
                 activation, 
                 kernel_initializer):
        super().__init__()
        
        self._input_transform = None
        self._output_transform = None
        
        if isinstance(activation, dict): # activation 설정
            activation_branch = activations.get(activation["branch"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)
            
        if callable(layer_sizes_branch[1]):
            self.branch = layer_sizes_branch[1]
        else:
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
            
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = nn.parameter.Parameter(torch.tensor(0.0))
    
    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        x_func = self.branch(x_func)
        
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x_loc = self.activation_trunk(self.trunk(x_loc))    
        
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk do not match."
            )
            
        x = torch.einsum("bi,bi->b", x_func, x_loc)
        
        x = torch.unsqueeze(x, 1)
        x += self.b
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
        
# mod = DeepONet([1,1,1,1], [1,1,1,1], "relu", modules.init_he)