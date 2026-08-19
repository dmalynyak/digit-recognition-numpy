# Digit Recognition from Scratch (NumPy)

A neural network for digit recognition, implemented from scratch using only NumPy
and trained on the MNIST_784 dataset, including a hyperparameter sweep and analysis of the results.

input image:  
<img src="image_examples/digit6.png" width="200" />  
prediction: 6
confidence: 97.7%

## Features
- Neural network implemented from scratch using NumPy
- Forward pass and backpropagation derived by hand
- Mini-batch gradient descent training
- ReLU and Softmax activation functions
- Categorical Cross-Entropy (CCE) loss
- Trained on the MNIST_784 dataset
- Hyperparameter experiments with statistics and analysis

## Project structure
```text
project/
├── train.py          # model definition and training
├── predict.py        # predict a digit from a given image
├── benchmarktests.py      # repeated-run benchmark and statistics
├── image_examples/   # example images for predict.py
├── models/           # saved models (.pkl)
└── README.md
```

## How it works

### Training
1. Splits MNIST_784 into two sets: 80% for training, 20% for accuracy evaluation.
2. Builds a 784–128–10 network (can be changed) with He initialization - weights are N(0,1) and multiplied by √(2 / number of in-weights).
3. Each epoch shuffles the data and splits it into mini-batches. For each batch runs a forward pass, then backpropagation using ReLU, Softmax, then error signal using CCE function.
4. Computes the gradients using error signals and takes one gradient-descent step per batch.
5. After training, evaluates accuracy on the rest (unused) 20% of the dataset.

### Recognition
1. Converts the input image to a 28×28 grayscale array, normalizes pixel values to [0, 1], and (possibly) inverts it so the digit matches MNIST.
2. Runs a forward pass through the trained model and returns the index of the highest-activation output neuron.

## Installation
```bash
git clone <>
cd project
pip install numpy scikit-learn pillow matplotlib
```

## Usage
```bash
python train.py                              # train with the configured hyperparameters and save to models/

python predict.py image_examples/image.png  # predict the digit in an image

python benchmarktests.py                          # run the repeated-run benchmark
```

## Results

Each configuration was trained 10 times from fresh random initialization. The table reports average values from the 10 runs.

| learning rate | epochs | batch size | avg. accuracy | avg. time (s) | std      |
|---------------|--------|------------|---------------|---------------|----------|
| 0.5           | 10     | 16         | 96.2%         | 6.0           | 0.003637 |
| 0.5           | 10     | 32         | 97.4%         | 3.9           | 0.002041 |
| 0.5           | 10     | 64         | 97.7%         | 3.5           | 0.001964 |
| 0.5           | 10     | 128        | 97.5%         | 3.0           | 0.000951 |
| 0.5           | 20     | 16         | 96.5%         | 12.4          | 0.001768 |
| 0.5           | 20     | 32         | 98.0%         | 7.1           | 0.000484 |
| 0.5           | 20     | 64         | 97.9%         | 6.4           | 0.000461 |
| 0.5           | 20     | 128        | 97.8%         | 5.5           | 0.000550 |
| 0.1           | 10     | 16         | 97.8%         | 8.9           | 0.001473 |
| 0.1           | 10     | 32         | 97.5%         | 6.4           | 0.000751 |
| 0.1           | 10     | 64         | 97.1%         | 5.5           | 0.000657 |
| 0.1           | 10     | 128        | 96.2%         | 4.7           | 0.001059 |
| 0.1           | 20     | 16         | 97.9%         | 17.0          | 0.001039 |
| 0.1           | 20     | 32         | 97.8%         | 11.8          | 0.000519 |
| 0.1           | 20     | 64         | 97.5%         | 11.0          | 0.000881 |
| 0.1           | 20     | 128        | 97.1%         | 8.9           | 0.000647 |
| 0.5           | 50     | 32         | 98.0%         | 29.7          | 0.000478 |
| 0.5           | 50     | 64         | 97.9%         | 26.5          | 0.000930 |
| 0.1           | 50     | 96         | 97.8%         | 28.9          | 0.000964 |
| 0.1           | 50     | 128        | 97.8%         | 32.6          | 0.000798 |
| 0.5           | 20     | full-batch | 81.9%         | 3.6           | 0.006732 |
| 0.5           | 50     | full-batch | 89.6%         | 7.4           | 0.004063 |
| 0.5           | 100    | full-batch | 91.7%         | 15.3          | 0.001863 |
| 0.1           | 100    | full-batch | 87.5%         | 15.5          | 0.001585 |


### Analysis

Several patterns emerge from the sweep:

**Mini-batch training converges faster than full-batch training.** Every configuration with mini-batching approach reaches 96–98% within 10–20 epochs in less than 10s, because each epoch performs hundreds of weight updates rather than the single one in full-batch gradient descent. Making **mini-batch algorithm approximately 9% more accurate** using the fraction of the time. 

**Batch size is directly linked with learning rate.** At the higher rate (η = 0.5), accuracy peaks around batch size 32–64, while the smallest batch (16) is both the least accurate and the slowest — gradient descent step is too "noisy" and perhaps can overshoot minimum. At the lower rate (η = 0.1) the pattern is opposite: smaller batches perform better, because the extra updates compensate for the smaller step size. **Learning rate and batch size are strictly dependent**.

**Accuracy plateaus.** Increasing epochs from 20 to 50 makes almost no improvement (η = 0.5, batch 32 stays at ~98.0%), so the network reaches the capacity of possible improvement of this architecture before 50 epochs. Training beyond that point only costs time.

**Variance is low.** Across 10 runs per configuration, the standard deviation stays below 0.004 so the results are stable depending on random initialization and shuffling, it reflects stability of setup depending on initial conditions.

**Best configuration:** η = 0.5, 20 epochs, batch size 32 — 98.0% accuracy in ~7.1s, with the low variance (std ≈ 0.0005). Batch size 64 is nearly identical, making either a reasonable default.
