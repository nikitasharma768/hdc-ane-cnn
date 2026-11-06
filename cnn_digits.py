"""
CNN baseline for the HDC-on-ANE comparison.

Trains a small VGG-style CNN on the scikit-learn digits dataset (same data and
same train/test split as Sri's HDCMultipleInference.py), then converts one
Core ML model per batch size (2 to 2048) for timing and energy measurement.

The model takes the same flattened 64-value input as the HDC model, so the
same Swift app and input data can be used to run both.
"""
import os
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import coremltools as ct

torch.manual_seed(42)
np.random.seed(42)

# --------------------------------------------------
# 1. Load dataset (identical to the HDC script)
# --------------------------------------------------
digits = load_digits()
X = digits.images.reshape(digits.images.shape[0], -1)  # flatten to 64
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# --------------------------------------------------
# 2. CNN model (VGG-style: stacked 3x3 convs + pooling)
# --------------------------------------------------
class CNNModel(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4 -> 2x2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, X):
        # Same input as HDC: (batch, 64). Reshape to an 8x8 image and
        # scale pixels (0-16) to 0-1 inside the model.
        x = X.reshape(-1, 1, 8, 8) / 16.0
        return self.classifier(self.features(x))


# --------------------------------------------------
# 3. Train
# --------------------------------------------------
model = CNNModel()
n_params = sum(p.numel() for p in model.parameters())
print(f"CNN parameters: {n_params:,}")

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
EPOCHS = 30
TRAIN_BATCH = 64

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_train))
    total_loss = 0.0
    for i in range(0, len(X_train), TRAIN_BATCH):
        idx = perm[i:i + TRAIN_BATCH]
        opt.zero_grad()
        loss = loss_fn(model(X_train[idx]), y_train[idx])
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(idx)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}  loss {total_loss / len(X_train):.4f}")

model.eval()
with torch.no_grad():
    acc = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
print(f"PyTorch CNN test accuracy: {acc:.4f}")
torch.save(model.state_dict(), "cnn_digits.pt")

# --------------------------------------------------
# 4. Convert to Core ML: one model per batch size
# --------------------------------------------------
OUT_DIR = "CNNModels"
os.makedirs(OUT_DIR, exist_ok=True)
input_size = X_train.shape[1]
batch_sizes = [2 ** k for k in range(1, 12)]  # 2, 4, 8, ..., 2048

for b in batch_sizes:
    example_input = torch.rand(b, input_size) * 16
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=(b, input_size))],
    )
    path = os.path.join(OUT_DIR, f"CNN_Batch{b}_Inference.mlpackage")
    mlmodel.save(path)
    print(f"Saved {path}")

    # Sanity check: Core ML model gives the same accuracy as PyTorch
    if b == 256:
        try:
            spec = mlmodel.get_spec()
            in_name = spec.description.input[0].name
            out_name = spec.description.output[0].name
            out = mlmodel.predict({in_name: X_test[:256].numpy()})[out_name]
            cm_acc = (np.argmax(out, axis=1) == y_test[:256].numpy()).mean()
            print(f"Core ML check (batch 256): accuracy {cm_acc:.4f}, "
                  f"input name '{in_name}', output name '{out_name}'")
        except Exception as e:
            print(f"Core ML check skipped: {e}")

print("Done. Models are in the CNNModels folder.")
