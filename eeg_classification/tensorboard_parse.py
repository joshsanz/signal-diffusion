from ast import literal_eval
import os
from os.path import join as pjoin
import shutil
import glob
import pprint
import traceback
import click
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sweep_path = "/home/abastani/signal-diffusion/eeg_classification/sweep_results.txt"

# tensor_clean(sweep_path)

def tensor_clean(sweep_path):
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


# Convert single tensorflow log file to pandas DataFrame
def tflog2pandas(path: str) -> pd.DataFrame:
    """
    Parameters
    ----------
    path : str
        path to tensorflow log file

    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def many_logs2pandas(event_paths):
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs

def event_log_to_dict(event_log_path: str) -> dict:
    log_panda = tflog2pandas(event_log_path)

def comment_to_dict(comment: str) -> dict:
    comment = comment.split(',')


def generate_hip_dicts(logs_path: str) -> dict:
    tensor_logs = os.listdir(logs_path)
    tensor_logs.sort()

    hip_dicts = []

    print("len: ", len(tensor_logs))

    for tensor_log in tensor_logs:
        log_comment = tensor_log
        event_log_fold_path = pjoin(logs_path, log_comment)
        event_log = os.listdir(event_log_fold_path)[0]
        event_log_path = pjoin(event_log, event_log_fold_path)
        # print(event_log_path)
