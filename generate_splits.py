import numpy as np
from sklearn.datasets import fetch_olivetti_faces

# Seed derived from roll number M26CSA012
roll_number = "M26CSA012"
seed = int(''.join([ch for ch in roll_number if ch.isdigit()]))

rng = np.random.default_rng(seed)

faces = fetch_olivetti_faces()
x = faces.data
y = faces.target

train_idx, val_idx, test_idx = [], [], []
for person_id in range(40):
    person_indices = np.where(y == person_id)[0]
    shuffled = rng.permutation(person_indices)
    train_idx.extend(shuffled[:6])
    val_idx.extend(shuffled[6:8])
    test_idx.extend(shuffled[8:10])
train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
test_idx = np.array(test_idx)

x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]
x_test, y_test = x[test_idx], y[test_idx]

# Save splits
np.savez(
    "olivetti_splits.npz",
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=y_test,
    x_test_masked=None  # placeholder, can be generated later if needed
)
print("Generated olivetti_splits.npz with shapes:")
print("x_train", x_train.shape)
print("x_val", x_val.shape)
print("x_test", x_test.shape)
