"""
General utils for training, evaluation and data loading
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, BatchSampler
from pathlib import Path


# https://github.com/xmed-lab/ECBM/blob/main/data/celeba.py
LABEL_ATTR_IDX = [2, 19, 20, 21, 31, 36, 18, 33]
N_ATTRIBUTES = 6
CONCEPT_ATTR_IDX = LABEL_ATTR_IDX[:N_ATTRIBUTES]
#LABEL_REMAP = tuple([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 102, 104, 105, 106, 108, 109, 110, 111, 112, 113, 114, 116, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 172, 173, 174, 175, 176, 177, 178, 180, 181, 182, 183, 184, 185, 186, 188, 189, 190, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 228, 230, 231, 232, 233, 234, 236, 237, 238, 239, 240, 241, 244, 248, 249, 250, 251, 252, 253, 254, 255])


class CelebADataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CelebA dataset
    """
    def __init__(
        self,
        root: list[Path | str],
        use_attr: bool,
        no_img: bool,
        uncertain_label: bool,
        image_dir: Path | str | None,
        n_class_attr: int,
        transform=None,
        excluded_attr_indexes=set()
    ) -> None:
        assert len(root) == 1
        #assert image_dir is None
        assert uncertain_label == False
        root = Path(root[0])
        self._root = root.parent
        split = {'train': 0, 'val': 1, 'test': 2}.get(root.stem)
        if split is None: raise ValueError('Unknown split', split)
        self._is_train = split == 0
        
        with open(self._root / 'list_eval_partition.txt', encoding='utf-8') as file:
            lines = file.readlines()

        names = list()
        for line in lines:
            assert line[-1] == '\n'
            fname, sp = line[:-1].split(' ')
            if int(sp) != split:
                continue
            names.append(fname)

        with open(self._root / 'list_attr_celeba.txt', encoding='utf-8') as file:
            lines = file.readlines()
        assert int(lines[0]) > 0
        lines.pop(0)
        attr_names = lines.pop(0).split()

        self.concept_names = tuple(map(lambda x: attr_names[x], CONCEPT_ATTR_IDX))

        name_to_attr = dict()
        for line in lines:
            cells = line.split()
            name = cells[0]
            attr = tuple(map(lambda x: 1 if x == '1' else 0, cells[1:]))
            name_to_attr[name] = attr

        concepts, labels = list(), list()
        for attr in map(lambda x: np.array(name_to_attr[x]), names):
            concept = attr[CONCEPT_ATTR_IDX]
            label = attr[LABEL_ATTR_IDX]
            sum, pow = 0, 1
            for bit in label[::-1]:
                if bit == 1: sum += pow
                pow *= 2
            concepts.append(concept)
            labels.append(sum)
        #label_remap = {v:i for i,v in enumerate(LABEL_REMAP)}
        #remapped_labels = list(map(lambda x: label_remap[x], labels))
        self.data = tuple(map(list, zip(names, concepts, labels)))
        self.label_count = 2**len(LABEL_ATTR_IDX)

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

        for idx in removed_idx[::-1]:
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
        #print(len(self))
        self.data = tuple(new_data[::12])

    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, index: int):
        img_path, attr_label, class_label = self.data[index]

        img = Image.open(self._root / 'img_align_celeba_png' / f'{img_path[:-4]}.png').convert('RGB')        
        if self.transform: img = self.transform(img)

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
    assert resol == 299
    is_training = any(['train.pkl' in f for f in pkl_paths])
    '''
    resized_resol = int(resol * 256/224)
    if is_training:
        transform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            #transforms.RandomSizedCrop(resol),
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            ])
    else:
        transform = transforms.Compose([
            #transforms.Resize((resized_resol, resized_resol)),
            transforms.CenterCrop(resol),
            transforms.ToTensor(), #implicitly divides by 255
            transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            ])
    '''
    transform=transforms.Compose([
        transforms.Resize(resol),
        transforms.CenterCrop(resol),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    generator = torch.Generator(torch.zeros(1).device)
    dataset = CelebADataset(pkl_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform, excluded_attr_indexes=excluded_attr_indexes)
    print(f'len({Path(pkl_paths[0]).stem})={len(dataset)}')

    if is_training:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False
    if resampling:
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
    dataset = CelebADataset([path], True, False, False, None, 2)
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
    root = 'D:/Workspace/Dataset/celeba/train.pkl'
    loader = load_data([root], True, False, 2, False, image_dir=None,
                       n_class_attr=2, resampling=False, excluded_attr_indexes=set())
    img, class_label, attr_label, index = next(iter(loader))
    imb = find_class_imbalance(root)
    print(imb)
