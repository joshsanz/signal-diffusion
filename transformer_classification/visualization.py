import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


ear_class_map = {0: "awake", 1: "sleepy"}
math_class_map = {0: "bkgnd_male", 1: "bkgnd_female", 2: "math_male", 3: "math_female"}


def class_confusion(model, data_loader, class_map, device='cpu'):
    model.eval()
    nclass = len(class_map)
    confusion = np.zeros((nclass, nclass), dtype=np.int32)
    for X, y in data_loader:
        with torch.no_grad():
            logits = model(X.to(device))
        y_hat = torch.argmax(logits, dim=-1)
        for true, pred in zip(y, y_hat):
            confusion[true, pred] += 1

    fig, _ = plt.subplots()
    sns.heatmap(confusion / np.sum(confusion, axis=0, keepdims=True), annot=True, fmt=".2%", cmap="Blues",
                xticklabels=class_map.values(), yticklabels=class_map.values())
    plt.yticks(rotation=60)
    plt.xlabel("Predicted Class\nAccuracy: {:.2%}".format(np.trace(confusion) / np.sum(confusion)))
    plt.ylabel("True Class")
    return confusion, fig


def class_prevalence(data_loader, class_map):

    nclass = len(class_map)
    counts = np.zeros(nclass, dtype=np.int32)
    for _, y in data_loader:
        for c in y:
            counts[c] += 1
    p = plt.figure()
    plt.xlabel("Class")
    plt.ylabel("Percent of Samples")
    plt.bar(class_map.values(), counts / np.sum(counts))
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))
    return p
