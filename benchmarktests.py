import numpy as np
import train
import time

def accuracy_test(X_train, X_test, Y_train, y_test, rep_num):
    accuacies = []
    times = []
    for i in range(rep_num):

        model = train.NeuralNetwork([784, 128, 10]) # making new network with random weights and biases
        start = time.perf_counter()
        model.learn_mini_batch(X_train, Y_train, eta=0.5, epochs=20, batch_size=32)
        end = time.perf_counter()
        elapsed = end - start
        accuracy = train.accuracy(model, X_test, y_test)
        accuacies.append(accuracy)
        times.append(elapsed)

    return accuacies, times

def main():

    X_train, X_test, Y_train, y_test = train.load_data()
    accuracies, times = accuracy_test(X_train, X_test, Y_train, y_test, rep_num=10)
    accuracies, times = np.array(accuracies), np.array(times)

    # visual arrays
    print("accuracies:", [f"{a:.3f}" for a in accuracies])
    print("times:", [f"{t:.3f}" for t in times])

    print(f"average accuracy: {np.mean(accuracies):.4f}, median {np.median(accuracies):.4f}, min: {np.min(accuracies):.4f} , max {np.max(accuracies):.4f}, std {np.std(accuracies):.6f}, variance {np.var(accuracies):.6f}")
    print(f"average time: {np.mean(times):.4f}, median {np.median(times):.4f}, min: {np.min(times):.4f} , max {np.max(times):.4f}, std {np.std(times):.6f}, variance {np.var(times):.6f}")

if __name__ == "__main__":
    main()