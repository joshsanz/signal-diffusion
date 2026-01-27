# Summary plots for generative results
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
                                   'Lucida Grande', 'Verdana']

# Load the data
classifier_acc_file = 'eeg_classification/bestmodels/accuracy.csv'
gen_metrics_sd_file = 'metrics/metrics-sdv1.json'
gen_metrics_dis_file = 'metrics/metrics-dis.json'
gen_metrics_self_file = 'metrics/metrics-self.json'
gen_metrics_sdxl_file = 'metrics/metrics-sdxl-vae.json'
gen_metrics_sdv1_file = 'metrics/metrics-sdv1-vae.json'
metrics_files = [
    gen_metrics_sd_file, gen_metrics_dis_file,
    gen_metrics_sdxl_file, gen_metrics_sdv1_file,
    gen_metrics_self_file,
]

classifier_acc = pd.read_csv(classifier_acc_file, index_col=0)
classifier_acc = classifier_acc.iloc[1:]
classifier_acc.rename_axis("Test", axis=1, inplace=True)
classifier_acc.rename_axis("Train", axis=0, inplace=True)
print(classifier_acc)
print(classifier_acc.values)

gen_metrics = pd.DataFrame()
for mf in metrics_files:
   df = pd.read_json(mf, orient='index')
   df["Generator"] = ["-".join(os.path.basename(mf).split("-")[1:]).split(".")[0]] * 2
   gen_metrics = pd.concat([gen_metrics, df])
gen_metrics = gen_metrics.reset_index(names="Featurizer")
gen_metrics = gen_metrics.replace("dinov2-vit-l-14", "DINOv2")
gen_metrics = gen_metrics.replace("clip-vit-l-14", "CLIP")
gen_metrics.set_index(["Generator", "Featurizer"], inplace=True)
print(gen_metrics)

# Plot the data
# Classifier accuracy
fig, ax = plt.subplots(figsize=(5, 8))
cax = ax.matshow(classifier_acc, cmap='viridis', clim=[0, 1])
plt.colorbar(cax)
plt.title("Classifier Accuracy")
plt.xlabel("Test")
plt.ylabel("Train")
plt.xticks(range(len(classifier_acc.columns)), classifier_acc.columns, rotation=45)
plt.yticks(range(len(classifier_acc.index)), classifier_acc.index)
plt.tight_layout()
# plt.show()
plt.savefig('classifier-accuracy-matrix.png')
plt.close()

fig, ax = plt.subplots(figsize=(12, 5))
# classifier_acc.plot(kind='bar', ax=ax)

x = np.arange(3)  # the label locations
width = 0.15  # the width of the bars
multiplier = 0

testsets = ['real_eeg', 'dis', 'sdv1']
trainsets = ['real_eeg', 'real_eeg EMA', 'dis', 'dis EMA', 'sdv1', 'sdv1 EMA']
colors = ['C0', 'C0', 'C1', 'C1', 'C2', 'C2']
alphas = [1, 0.5, 1, 0.5, 1, 0.5]
for trainset, c, a in zip(trainsets, colors, alphas):
    offset = width * multiplier
    rects = ax.bar(x + offset, classifier_acc.loc[trainset], width, label=trainset, color=c, alpha=a)
    # ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(x + width * 2.5, testsets)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.15), ncols=3, title='Training Set [Model]')
ax.set_ylim(0, 1.05)

# plt.xticks(rotation=45)
plt.xlabel("Test Set")
plt.ylabel("Test Set Accuracy")
plt.grid()
# plt.title("Classifier Accuracy")
plt.tight_layout()
# plt.show()
plt.savefig('classifier-accuracy-bar.png')
plt.close()

# Generative Metrics
# KDD
fig, ax = plt.subplots(figsize=(8, 5))
gen_metrics.unstack().plot.bar(y="kernel_inception_distance_mean", ax=ax,
                               yerr="kernel_inception_distance_std", capsize=5)
plt.xticks(rotation=0)
plt.ylabel(r"MMD$\downarrow$")
# plt.title("Maximal Mean Discrepancy (MMD)")
plt.tight_layout()
# plt.show()
plt.savefig('kdd-bar.png')
plt.close()
