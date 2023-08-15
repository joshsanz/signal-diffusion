from ast import literal_eval
import os
from os.path import join as pjoin
import shutil

sweep_path = "/home/abastani/signal-diffusion/eeg_classification/sweep_results.txt"

with open(sweep_path) as f:
    lines = f.readlines()

sweeps = [literal_eval(sweep) for sweep in lines if sweep[0] != '#']
sweep_comments = [sweep[2] for sweep in sweeps]
sweep_comments_set = set(sweep_comments)

tensorboard_path = '/home/abastani/signal-diffusion/eeg_classification/tensorboard_logs/cnn'

tensor_logs = os.listdir(tensorboard_path)
tensor_logs.sort()

print("PRE tensor size: ", len(tensor_logs))

for log in tensor_logs:
    if log.split('-')[0] not in sweep_comments_set:
        shutil.rmtree(pjoin(tensorboard_path, log))

print("POST tensor size: ", len(os.listdir(tensorboard_path)))
print("sweep len: ", len(sweep_comments_set))
# comment = sweep_comments[0]
# comment_path = pjoin(tensorboard_path, comment)
# print(comment_path)
# print("/home/abastani/signal-diffusion/eeg_classification/tensorboard_logs/cnn/run_3: cnnclass_CNNClassifierLight_SGD_decay0.1_epoch:5,swa_start:3,base_lr:0.1,swa_lr:0.01,epsilon:0.5,sched:None-2023-08-02_13:09:53.310691")
# if os.path.isdir('/home/abastani/signal-diffusion/eeg_classification/tensorboard_logs/cnn/run_3: cnnclass_CNNClassifierLight_SGD_decay0.1_epoch:5,swa_start:3,base_lr:0.1,swa_lr:0.01,epsilon:0.5,sched:None-2023-08-02_13:09:53.310691'):
#     print("alsj")
