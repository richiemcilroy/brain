# data/

The continual-learning experiments read MNIST from `data/mnist.npz`. The file is
**not committed** (it is 11 MB of third-party data), so fetch it once:

```bash
curl -sL -o data/mnist.npz \
  https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
```

Verify it after download — the loader expects these exact keys:

```bash
python3 -c "import numpy as np; d=np.load('data/mnist.npz'); \
print(sorted(d.keys()), d['x_train'].shape, d['x_train'].dtype)"
# -> ['x_test', 'x_train', 'y_test', 'y_train'] (60000, 28, 28) uint8
```

The file is the standard Keras MNIST mirror (`x_train` uint8 `(60000,28,28)`,
`y_train` uint8 `(60000,)`, plus test splits). Nothing else in this repo
downloads anything at runtime — see `brain/tasks.py:load_mnist`, which reads the
local file and raises `FileNotFoundError` with a resume hint if it is missing
rather than silently reaching for the network.

`experiments/xor_neuron.py`, `bench/bench_scale.py`, and the whole `tests/`
suite run **without** this file; only the continual-learning experiments need it.

SHA256 of the mirror used in this project:

```bash
shasum -a 256 data/mnist.npz
```

Recorded value for the copy used for the results in `docs/RESULTS.md`:
`731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`
