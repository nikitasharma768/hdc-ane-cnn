import torch
from torch import nn
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import coremltools as ct

# --------------------------
# 1️⃣ Load dataset
# --------------------------
digits = load_digits()
X = digits.images.reshape(digits.images.shape[0], -1)  # flatten to 64
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# --------------------------
# 2️⃣ HDC Model (PyTorch)
# --------------------------
class HDCModel(nn.Module):
    def __init__(self, input_size, D, n_classes):
        super().__init__()
        self.D = D
        self.n_classes = n_classes
        
        # Random encoding matrix
        self.register_buffer("encoding_matrix", torch.tensor(
            np.random.choice([-1, 1], size=(D, input_size)), dtype=torch.float32
        ))
        self.class_hvs = torch.zeros(D, n_classes)
    
    def train_hd(self, X_train, y_train):
        for x, label in zip(X_train, y_train):
            hv = (self.encoding_matrix @ x >= 0).float() * 2 - 1  # thresholding instead of sign
            self.class_hvs[:, label] += hv
        self.class_hvs = (self.class_hvs >= 0).float() * 2 - 1
    
    def forward(self, X):
        hv = (X @ self.encoding_matrix.T >= 0).float() * 2 - 1
        similarities = hv @ self.class_hvs
        return similarities
    
    def predict(self, X):
        sims = self.forward(X)
        return torch.argmax(sims, dim=1)

# --------------------------
# 3️⃣ Instantiate and train
# --------------------------
D = 80000
n_classes = 10
input_size = X_train.shape[1]

hdc_model = HDCModel(input_size, D, n_classes)
hdc_model.train_hd(X_train, y_train)

# --------------------------
# 4️⃣ PyTorch Batch Inference
# --------------------------
batch_size = 2048
y_pred_list = []

for i in range(0, len(X_test), batch_size):
    X_batch = X_test[i:i+batch_size]
    y_pred_batch = hdc_model.predict(X_batch)
    y_pred_list.append(y_pred_batch)

y_pred = torch.cat(y_pred_list)
accuracy = (y_pred == y_test).float().mean().item()
print("PyTorch HDC Accuracy (batch=256):", accuracy)

# --------------------------
# 5️⃣ Convert to CoreML (batch=256)
# --------------------------
example_input = torch.randn(batch_size, input_size)
traced_model = torch.jit.trace(hdc_model, example_input)

model_from_trace = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=(batch_size, input_size))],
)

model_from_trace.save("/Users/srivatsansureshbabu/Downloads/CNN_V4_GPU_8/DifferentDimensionalitiesBatch2048/Dim80000Batch2048_Inference.mlpackage")
print("CoreML model saved as hdc_model_batch256.mlpackage")
