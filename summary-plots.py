# Summary plots for generative results

import matplotlib.pyplot as plt
import pandas as pd
import os


# Load the data
classifier_acc_file = 'eeg_classification/bestmodels/accuracy.csv'
gen_metrics_sd_file = 'metrics/metrics-sd.json'
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

fig, ax = plt.subplots(figsize=(10, 5))
classifier_acc.plot(kind='bar', ax=ax)
plt.xticks(rotation=45)
plt.xlabel("Train")
plt.ylabel("Test Set Accuracy")
plt.grid()
plt.title("Classifier Accuracy")
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
plt.title("Maximal Mean Discrepancy (MMD)")
plt.tight_layout()
# plt.show()
plt.savefig('kdd-bar.png')
plt.close()
