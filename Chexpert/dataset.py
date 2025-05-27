"""
General utils for training, evaluation and data loading
"""
import os
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd


CSV_COLUMNS = list('Path,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices'.split(','))
N_ATTRIBUTES = 13


class ChexpertDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CelebA dataset
    """
    def __init__(
        self,
        paths: list[Path | str],
        use_attr: bool,
        no_img: bool,
        uncertain_label: bool,
        image_dir: Path | str | None,
        n_class_attr: int,
        transform=None,
        excluded_attr_indexes=set()
    ) -> None:
        assert uncertain_label == False
        self._root = Path(paths[0]).parent
        self._is_train = False
        self.data = list()

        for path in paths:
            path = Path(path)
            split = {'train': 'train', 'val': 'valid', 'test': 'test'}.get(path.stem)
            if split is None: raise ValueError('Unknown split', path.stem)
            self._is_train |= split == 'train'
            dataset = pd.read_csv(self._root / split / 'labels.csv')

            names = dataset['Path'].to_numpy().tolist()
            labels = dataset['No Finding'].to_numpy().astype(int).tolist()
            concepts = dataset[CSV_COLUMNS[2:]].to_numpy().tolist()
            for name, concept, label in zip(names, concepts, labels):
                name = '/'.join(name.split('/')[-4:])
                if (self._root / name).exists():
                    self.data.append([name, concept, int(label)])

        self.data = tuple(self.data)
        if self._is_train: self.data = self.data[::10]

        self._excluded_attr = excluded_attr_indexes.copy()
        self._remove_data = None
        self._remove_data_cnt = 0
        self._remove_data_is_single = None

        for k in self._excluded_attr:
            if not isinstance(k, str):
                continue
            assert k.startswith('REMOVE_DATA')
            self._excluded_attr.remove(k)
            self.valid_attr_indexes = list(range(N_ATTRIBUTES))
            self._remove_data_cnt = k[len('REMOVE_DATA_'):]

            if self._remove_data_cnt == '':
                self._remove_data_cnt = 0
                assert len(self._excluded_attr) == 1    
                self._remove_data = self._excluded_attr.pop()
            else:
                assert len(self._excluded_attr) == 0
                idx = self._remove_data_cnt.find('S')
                if idx != -1:
                    self._remove_data_is_single = self._remove_data_cnt[idx+1:]
                    if self._remove_data_is_single == '': self._remove_data_is_single = -1
                    self._remove_data_is_single = int(self._remove_data_is_single)
                    self._remove_data_cnt = self._remove_data_cnt[:idx]
                self._remove_data_cnt = (float if '.' in self._remove_data_cnt else int)(self._remove_data_cnt)
            break
        else:
            self.valid_attr_indexes = list(sorted(set(range(N_ATTRIBUTES)) - self._excluded_attr))  # Index of the attribute to exclude
            assert len(self.valid_attr_indexes) == N_ATTRIBUTES - len(self._excluded_attr)

        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr

        new_data = list()
        if isinstance(self._remove_data_cnt, float):
            assert 0 < self._remove_data_cnt < 1
            cnt = int(self._remove_data_cnt * len(self.data))
        else:
            cnt = self._remove_data_cnt
        if self._is_train:
            removed_idx = np.sort(np.random.choice(len(self.data), cnt, replace=False))
        else:
            removed_idx = list()

        key = 1
        for data in self.data:
            attr_label = torch.tensor(data[key])[self.valid_attr_indexes]
            if self._is_train and self._remove_data is not None and attr_label[self._remove_data] > 0:
                continue
            data[key] = attr_label
            new_data.append(data)

        for idx in (removed_idx[::-1]):
            if self._remove_data_is_single is None:
                new_data.pop(idx)
                continue
            assert self._remove_data_is_single == -1
            if self._remove_data_is_single == -1:
                assert len(new_data[idx][key].shape) == 1
                valid_attr_idx = torch.where(new_data[idx][key])[0].cpu().tolist()
                if len(valid_attr_idx) == 0: continue
                attr_idx = np.random.choice(valid_attr_idx, 1)
                new_data[idx][key][attr_idx] = 1 - new_data[idx][key][attr_idx]
            else:
                new_data[idx][key][self._remove_data_is_single] == 1 - new_data[idx][key][self._remove_data_is_single]

        self.data = tuple(new_data)

    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, index: int):
        img, attr_label, class_label = self.data[index]
        img = Image.open(self._root / img).convert('RGB')

        if self.transform: img = self.transform(img)
        #print(img.shape)
        if self.use_attr:
            #attr_label = torch.tensor(attr_label)[self.valid_attr_indexes]

            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((N_ATTRIBUTES, self.n_class_attr))
                    one_hot_attr_label[np.arange(N_ATTRIBUTES), attr_label] = 1
                    return one_hot_attr_label, class_label
                else:
                    return attr_label, class_label
            else:
                return img, class_label, attr_label, index
        else:
            return img, class_label


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return dataset.data[idx]['attribute_label'][0]

    def __iter__(self):
        idx = (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
        return idx

    def __len__(self):
        return self.num_samples

def load_data(pkl_paths, use_attr, no_img, batch_size, uncertain_label=False, n_class_attr=2, image_dir='images', resampling=False, resol=299, excluded_attr_indexes=set()):
    """
    Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
    Loads data with transformations applied, and upsample the minority class if there is class imbalance and weighted loss is not used
    NOTE: resampling is customized for first attribute only, so change sampler.py if necessary
    """
    is_training = any(['train.pkl' in f for f in pkl_paths])
    if is_training:
        transform = transforms.Compose([
            #transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])
    else:
        transform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            #transforms.CenterCrop(resol),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])

    generator = torch.Generator(torch.zeros(1).device)
    dataset = ChexpertDataset(pkl_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform, excluded_attr_indexes=excluded_attr_indexes)
    print(f'len({Path(pkl_paths[0]).stem})={len(dataset)}')
    if is_training:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False
    if resampling and False:
        sampler = BatchSampler(ImbalancedDatasetSampler(dataset), batch_size=batch_size, drop_last=drop_last)
        loader = DataLoader(dataset, batch_sampler=sampler, generator=generator, num_workers=8, persistent_workers=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, generator=generator, num_workers=8, persistent_workers=True)
    return loader

def find_class_imbalance(path, multiple_attr=False, attr_idx=-1):
    """
    Calculate class imbalance ratio for binary attribute labels stored in pkl_file
    If attr_idx >= 0, then only return ratio for the corresponding attribute id
    If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
    """
    dataset = ChexpertDataset([path], True, False, False, None, 2)
    imbalance_ratio = []
    n = len(dataset)
    n_attr = len(dataset.data[0][1])
    if attr_idx >= 0:
        n_attr = 1
    if multiple_attr:
        n_ones = [0] * n_attr
        total = [n] * n_attr
    else:
        n_ones = [0]
        total = [n * n_attr]
    for d in dataset.data:
        labels = d[1]
        if multiple_attr:
            for i in range(n_attr):
                n_ones[i] += labels[i]
        else:
            if attr_idx >= 0:
                n_ones[0] += labels[attr_idx]
            else:
                n_ones[0] += sum(labels)
    for j in range(len(n_ones)):
        imbalance_ratio.append(total[j]/n_ones[j] - 1)
    if not multiple_attr: #e.g. [9.0] --> [9.0] * 312
        imbalance_ratio *= n_attr
    return imbalance_ratio

if __name__ == '__main__':
    loader = load_data(['/hpc2hdd/home/hlin199/Dataset/celeba/train.pkl'], True, False, 2, False, image_dir=None,
                       n_class_attr=2, resampling=False, excluded_attr_indexes=set())
    img, class_label, attr_label, index = next(iter(loader))
    imb = find_class_imbalance(loader.dataset)
    print(imb)
