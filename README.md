# CNN vs. HDC on Apple Neural Engines

Benchmarks a VGG-style CNN on Apple devices (CPU, GPU, Neural Engine) to compare
its speed and energy against hyperdimensional computing (HDC) results from
Suresh Babu, Shahryar, and Morris (2026).

**Student:** Nikita Sharma, Grant Esh· **Faculty mentor:** Dr. Justin Morris · **Student mentor:** Srivatsan Suresh Babu
California State University San Marcos

## Pipeline

1. **Train** a CNN in PyTorch (`cnn_digits.py`).
2. **Convert** it to Core ML with coremltools, one model per batch size (2 to 2048).
3. **Time** each model with a Swift benchmark (`bench.swift`): 20 runs over the test set.
4. **Measure power** with `powermetrics` while the benchmark runs (`run_power.sh`).
5. **Energy** = power × average inference time.

## Files

| File | What it does |
| --- | --- |
| `cnn_digits.py` | Trains the CNN on scikit-learn digits and saves Core ML models to `CNNModels/` |
| `export_digits.py` | Saves the digits test set to `digits_test.data` for the Swift benchmark |
| `bench.swift` | Command-line version of Sri's Swift benchmark (no Xcode needed) |
| `run_power.sh` | Runs every batch size on CPU, GPU, and ANE while recording power |
| `reference/sri/` | Sri's original HDC scripts, kept for reference |
| `results/` | Timing and power logs (created by `run_power.sh`) |

## Setup (Mac)

```
conda create -n hdc python=3.11 -y
conda activate hdc
pip install torch==2.7.0 torchvision==0.22.0 coremltools scikit-learn numpy pandas matplotlib
```

## Usage

```
python cnn_digits.py
python export_digits.py
swiftc -O bench.swift -o bench
./bench CNNModels/CNN_Batch256_Inference.mlpackage digits_test.data 256 ane
./run_power.sh   # needs sudo access to /usr/bin/powermetrics
```

Trained models (`CNNModels/`, `*.pt`) and datasets are not stored in the repo.
Re-create them with the scripts above.

## Status

| Device | Digits | MNIST | CIFAR-10 | CIFAR-100 |
| --- | --- | --- | --- | --- |
| Mac mini | Timing done, power pending | To do | To do | To do |
| iPhone | To do | To do | To do | To do |
| iPad | To do | To do | To do | To do |
| Apple Watch | To do | To do | To do | To do |
