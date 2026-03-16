import torch
import torch.nn as nn
import torch.optim as optim
import dump_utils as dump

# 1. Setup for reproducibility
torch.manual_seed(42)

# Simple dimensions: 3 inputs, 2 outputs
in_features = 3
out_features = 2
learning_rate = 0.1
weight_decay = 0.01

# 2. Define Model and Optimizer
# We'll use a standard Linear layer (y = xA^T + b)
model = nn.Linear(in_features, out_features)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 3. Create dummy input and target
# Using fixed values for easier manual verification if needed
input_tensor = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
target = torch.tensor([[0.5, -0.5]])

# Capture INITIAL weights and biases
w_initial = model.weight.detach().clone()
b_initial = model.bias.detach().clone()

# 4. Forward and Backward pass
output = model(input_tensor)
loss = nn.MSELoss()(output, target)
optimizer.zero_grad()
loss.backward()

# Capture GRADIENTS
w_grad = model.weight.grad.detach().clone()
b_grad = model.bias.grad.detach().clone()

# 5. Optimizer Step
optimizer.step()

# Capture UPDATED weights and biases
w_final = model.weight.detach().clone()
b_final = model.bias.grad.detach().clone() # Wait, this should be weight value, not grad
w_final = model.weight.detach().clone()
b_final = model.bias.detach().clone()

# 6. Display Results
def write_vec(f, name, tensor):
    f.write("static float " + name + "[] = {" + dump.tensor_to_string(tensor) + "};\n")

print("--- SGD TEST VECTORS ---")
print(f"LearningdRate: {learning_rate}\n")
with open("data.h", "w") as f:
    write_vec(f, "WEIGHTS_BEFORE", w_initial);
    write_vec(f, "BIASES_BEFORE", b_initial);
    write_vec(f, "WEIGHTS_AFTER", w_final);
    write_vec(f, "BIASES_AFTER", b_final);
    write_vec(f, "WEIGHTS_GRAD", w_grad);
    write_vec(f, "BIASES_GRAD", b_grad);
