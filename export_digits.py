"""
Exports the scikit-learn digits TEST set (same split as the HDC and CNN
scripts) to digits_test.data, in the same format as Sri's isolet5.data:
one sample per line, comma-separated features, label last (1-indexed).
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.images.reshape(digits.images.shape[0], -1)
y = digits.target

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with open("digits_test.data", "w") as f:
    for row, label in zip(X_test, y_test):
        values = ",".join(f"{v:.1f}" for v in row)
        f.write(f"{values},{label + 1}\n")  # +1 so labels match ISOLET's 1-indexing

print(f"Wrote {len(X_test)} samples with {X_test.shape[1]} features to digits_test.data")
