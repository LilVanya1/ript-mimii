#!/usr/bin/env python
"""
 @file   baseline.py
 @brief  Baseline code of simple AE-based anomaly detection (PyTorch port).
 @author Based on work by Ryo Tanabe and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2019 Hitachi, Ltd. All right reserved.
"""
import pickle
import os
import sys
import glob

import numpy as np
import librosa
import librosa.core
import librosa.feature
import yaml
import logging
from tqdm import tqdm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

__version__ = "1.0.3 (PyTorch port)"

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class Visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Test"], loc="upper right")

    def save_figure(self, name):
        self.plt.savefig(name)


def save_pickle(filename, save_data):
    logger.info("save_pickle -> {}".format(filename))
    with open(filename, 'wb') as sf:
        pickle.dump(save_data, sf)


def load_pickle(filename):
    logger.info("load_pickle <- {}".format(filename))
    with open(filename, 'rb') as lf:
        return pickle.load(lf)


def file_load(wav_name, mono=False):
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


def demux_wav(wav_name, channel=0):
    try:
        multi_channel_data, sr = file_load(wav_name)
        if multi_channel_data.ndim <= 1:
            return sr, multi_channel_data
        return sr, np.array(multi_channel_data)[channel, :]
    except ValueError as msg:
        logger.warning(f'{msg}')


def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    dims = n_mels * frames
    sr, y = demux_wav(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1
    if vectorarray_size < 1:
        return np.empty((0, dims), float)
    vectorarray = np.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T
    return vectorarray


def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    dims = n_mels * frames
    if len(file_list) == 0:
        return np.empty((0, dims), float)
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = file_to_vector_array(file_list[idx],
                                            n_mels=n_mels, frames=frames,
                                            n_fft=n_fft, hop_length=hop_length, power=power)
        if idx == 0:
            dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array
    return dataset


def dataset_generator(target_dir,
                      normal_dir_name="normal",
                      abnormal_dir_name="abnormal",
                      ext="wav"):
    logger.info("target_dir : {}".format(target_dir))
    normal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                  normal_dir_name=normal_dir_name, ext=ext))))
    normal_labels = np.zeros(len(normal_files))
    if len(normal_files) == 0:
        logger.exception("no_wav_data!!")

    abnormal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                    abnormal_dir_name=abnormal_dir_name, ext=ext))))
    abnormal_labels = np.ones(len(abnormal_files))
    if len(abnormal_files) == 0:
        logger.exception("no_wav_data!!")

    train_files = normal_files[len(abnormal_files):]
    train_labels = normal_labels[len(abnormal_files):]
    eval_files = np.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
    eval_labels = np.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)
    logger.info("train_file num : {num}".format(num=len(train_files)))
    logger.info("eval_file  num : {num}".format(num=len(eval_files)))

    return train_files, train_labels, eval_files, eval_labels


class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_size=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, bottleneck_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VAE(nn.Module):
    def __init__(self, input_dim, bottleneck_size=16):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.enc_fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mean = nn.Linear(64, bottleneck_size)
        self.log_var = nn.Linear(64, bottleneck_size)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def encode(self, x):
        h = self.enc_fc(x)
        return self.mean(h), self.log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def kl_loss(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)


def audio_to_patches(y, sr, n_mels=64, patch_frames=64, stride=32,
                     n_fft=1024, hop_length=512, power=2.0):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)
    log_mel = 20.0 / power * np.log10(mel + sys.float_info.epsilon)
    T = log_mel.shape[1]
    if T < patch_frames:
        return np.empty((0, 1, n_mels, patch_frames), float)
    patches = []
    for t in range(0, T - patch_frames + 1, stride):
        patches.append(log_mel[:, t:t + patch_frames])
    if not patches:
        return np.empty((0, 1, n_mels, patch_frames), float)
    arr = np.array(patches)[:, np.newaxis, :, :]
    return arr.astype(np.float32)


def file_to_patches(file_name, n_mels=64, patch_frames=64, stride=32,
                    n_fft=1024, hop_length=512, power=2.0):
    sr, y = demux_wav(file_name)
    return audio_to_patches(y, sr, n_mels=n_mels, patch_frames=patch_frames,
                            stride=stride, n_fft=n_fft, hop_length=hop_length, power=power)


def list_to_patch_array(file_list, msg="patches...",
                        n_mels=64, patch_frames=64, stride=32,
                        n_fft=1024, hop_length=512, power=2.0):
    all_patches = []
    for f in tqdm(file_list, desc=msg):
        patches = file_to_patches(f, n_mels=n_mels, patch_frames=patch_frames,
                                  stride=stride, n_fft=n_fft, hop_length=hop_length, power=power)
        if patches.shape[0] > 0:
            all_patches.append(patches)
    if not all_patches:
        return np.empty((0, 1, n_mels, patch_frames), float)
    return np.concatenate(all_patches, axis=0)


class CNNAutoencoder(nn.Module):
    def __init__(self, in_channels=1, base_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 5, stride=2, padding=2),
            nn.BatchNorm2d(base_dim), nn.ReLU(),
            nn.Conv2d(base_dim, base_dim*2, 5, stride=2, padding=2),
            nn.BatchNorm2d(base_dim*2), nn.ReLU(),
            nn.Conv2d(base_dim*2, base_dim*4, 5, stride=2, padding=2),
            nn.BatchNorm2d(base_dim*4), nn.ReLU(),
            nn.Conv2d(base_dim*4, base_dim*8, 5, stride=2, padding=2),
            nn.BatchNorm2d(base_dim*8), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_dim*8, base_dim*4, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(base_dim*4), nn.ReLU(),
            nn.ConvTranspose2d(base_dim*4, base_dim*2, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(base_dim*2), nn.ReLU(),
            nn.ConvTranspose2d(base_dim*2, base_dim, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(base_dim), nn.ReLU(),
            nn.ConvTranspose2d(base_dim, in_channels, 5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.
    for batch_x in loader:
        batch_x = batch_x[0].to(device)
        recon = model(batch_x)
        loss = criterion(recon, batch_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x[0].to(device)
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            total_loss += loss.item() * batch_x.size(0)
    return total_loss / len(loader.dataset)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)

    os.makedirs(param["pickle_directory"], exist_ok=True)
    os.makedirs(param["model_directory"], exist_ok=True)
    os.makedirs(param["result_directory"], exist_ok=True)

    visualizer = Visualizer()

    base = os.path.abspath(param["base_directory"])
    dirs = []
    for root, subdirs, _ in os.walk(base):
        if 'normal' in subdirs and 'abnormal' in subdirs:
            dirs.append(root)
            subdirs[:] = [d for d in subdirs if d not in ('normal', 'abnormal')]
    dirs.sort()

    result_file = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
    results = {}

    for dir_idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(dirs)))

        rel = os.path.relpath(target_dir, base).replace(os.sep, '/')
        parts = rel.split('/')
        if len(parts) >= 3:
            db = parts[-3]
            machine_type = parts[-2]
            machine_id = parts[-1]
        elif len(parts) == 2:
            machine_id = parts[-1]
            combined = parts[-2]
            machine_type = combined.split('_')[-1]
            db = '_'.join(combined.split('_')[:-1]) if '_' in combined else combined
        else:
            machine_id = parts[-1]
            machine_type = machine_id
            db = "unknown"

        evaluation_result = {}
        key = "{machine_type}_{machine_id}_{db}".format(machine_type=machine_type, machine_id=machine_id, db=db)
        train_pickle = "{pickle}/train_{key}.pickle".format(pickle=param["pickle_directory"], key=key)
        eval_files_pickle = "{pickle}/eval_files_{key}.pickle".format(pickle=param["pickle_directory"], key=key)
        eval_labels_pickle = "{pickle}/eval_labels_{key}.pickle".format(pickle=param["pickle_directory"], key=key)
        scaler_pickle = "{pickle}/scaler_{key}.pickle".format(pickle=param["pickle_directory"], key=key)
        model_file = "{model}/model_{key}.pth".format(model=param["model_directory"], key=key)
        history_img = "{model}/history_{machine_type}_{machine_id}_{db}.png".format(
            model=param["model_directory"], machine_type=machine_type, machine_id=machine_id, db=db)
        print("============== DATASET_GENERATOR ==============")
        if os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
            train_data = load_pickle(train_pickle)
            eval_files = load_pickle(eval_files_pickle)
            eval_labels = load_pickle(eval_labels_pickle)
        else:
            train_files, train_labels, eval_files, eval_labels = dataset_generator(target_dir)
            train_data = list_to_vector_array(train_files,
                                              msg="generate train_dataset",
                                              n_mels=param["feature"]["n_mels"],
                                              frames=param["feature"]["frames"],
                                              n_fft=param["feature"]["n_fft"],
                                              hop_length=param["feature"]["hop_length"],
                                              power=param["feature"]["power"])
            save_pickle(train_pickle, train_data)
            save_pickle(eval_files_pickle, eval_files)
            save_pickle(eval_labels_pickle, eval_labels)

        print("============== MODEL TRAINING ==============")
        input_dim = param["feature"]["n_mels"] * param["feature"]["frames"]
        bottleneck_size = param["fit"].get("bottleneck_size", 16)
        model = Autoencoder(input_dim, bottleneck_size=bottleneck_size).to(device)
        logger.info(model)

        use_norm = param["fit"].get("normalize", False)
        if os.path.exists(model_file) and (not use_norm or os.path.exists(scaler_pickle)):
            model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
            model.to(device)
            scaler = load_pickle(scaler_pickle) if use_norm else None
        else:
            if use_norm:
                scaler = StandardScaler()
                train_data = scaler.fit_transform(train_data)
                save_pickle(scaler_pickle, scaler)
            else:
                scaler = None

            denoising_std = param["fit"].get("denoising_std", 0.0)
            train_tensor = torch.from_numpy(train_data).float()
            val_split = param["fit"].get("validation_split", 0.1)
            val_len = int(len(train_tensor) * val_split)
            train_len = len(train_tensor) - val_len
            train_data_tensor, val_data_tensor = random_split(train_tensor, [train_len, val_len])

            train_loader = DataLoader(train_data_tensor, batch_size=param["fit"]["batch_size"],
                                      shuffle=param["fit"]["shuffle"])
            val_loader = DataLoader(val_data_tensor, batch_size=param["fit"]["batch_size"],
                                    shuffle=False)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=param["fit"].get("learning_rate", 0.001))

            train_losses = []
            val_losses = []
            epochs = param["fit"]["epochs"]
            verbose = param["fit"].get("verbose", 1)

            for epoch in range(epochs):
                model.train()
                total_loss = 0.
                for batch_x in train_loader:
                    batch_x = batch_x.to(device)
                    if denoising_std > 0:
                        noisy_x = batch_x + torch.randn_like(batch_x) * denoising_std
                    else:
                        noisy_x = batch_x
                    recon = model(noisy_x)
                    loss = criterion(recon, batch_x)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * batch_x.size(0)
                train_loss = total_loss / len(train_loader.dataset)

                model.eval()
                total_val = 0.
                with torch.no_grad():
                    for batch_x in val_loader:
                        batch_x = batch_x.to(device)
                        recon = model(batch_x)
                        loss = criterion(recon, batch_x)
                        total_val += loss.item() * batch_x.size(0)
                val_loss = total_val / len(val_loader.dataset)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                if verbose:
                    logger.info(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

            visualizer.loss_plot(train_losses, val_losses)
            visualizer.save_figure(history_img)
            torch.save(model.state_dict(), model_file)

        print("============== EVALUATION ==============")
        model.eval()
        score_mode = param["fit"].get("score_mode", "mean")
        y_pred = [0. for k in eval_labels]
        y_true = eval_labels

        for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):
            try:
                data = file_to_vector_array(file_name,
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"])
                if scaler is not None:
                    data = scaler.transform(data)
                data_tensor = torch.from_numpy(data).float().to(device)
                with torch.no_grad():
                    recon = model(data_tensor)
                    frame_errors = torch.mean((data_tensor - recon) ** 2, dim=1).cpu().numpy()
                if score_mode == "max":
                    y_pred[num] = float(np.max(frame_errors))
                elif score_mode == "p95":
                    y_pred[num] = float(np.percentile(frame_errors, 95))
                else:
                    y_pred[num] = float(np.mean(frame_errors))
            except:
                logger.warning("File broken!!: {}".format(file_name))

        score = metrics.roc_auc_score(y_true, y_pred)
        logger.info("AUC : {}".format(score))
        evaluation_result["AUC"] = float(score)
        results[key] = evaluation_result
        print("===========================")

    print("\n===========================")
    logger.info("all results -> {}".format(result_file))
    with open(result_file, "w") as f:
        f.write(yaml.dump(results, default_flow_style=False))
    print("===========================")
