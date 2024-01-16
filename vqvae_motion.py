import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm


class VQMotionDataset(data.Dataset):
    def __init__(self, window_size=64, unit_length=4, debug=False):
        self.window_size = window_size
        self.unit_length = unit_length

        self.data_root = '/media/varora/LaCie/Datasets/HumanML3D/HumanML3D/'
        self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
        self.text_dir = pjoin(self.data_root, 'texts')
        self.joints_num = 22
        self.max_motion_length = 196
        self.meta_dir = '/media/varora/LaCie/Datasets/HumanML3D/HumanML3D/'

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))

        split_file = pjoin(self.data_root, 'train.txt')

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        if debug:
            id_list = id_list[:200]

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if motion.shape[0] < self.window_size:
                    continue
                self.lengths.append(motion.shape[0] - self.window_size)
                self.data.append(motion)
            except:
                pass  # missing motion

        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def compute_sampling_prob(self):
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]

        idx = random.randint(0, len(motion) - self.window_size)

        motion = motion[idx:idx + self.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


def motion_dataloader(
               batch_size,
               num_workers=8,
               window_size=64,
               unit_length=4,
               debug=False
):
    trainSet = VQMotionDataset(window_size=window_size, unit_length=unit_length, debug=debug)
    prob = trainSet.compute_sampling_prob()
    sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples=len(trainSet) * 1000, replacement=True)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                               batch_size,
                                               shuffle=True,
                                               # sampler=sampler,
                                               num_workers=num_workers,
                                               # collate_fn=collate_fn,
                                               drop_last=True)
    return train_loader

if '__main__' == __name__:
    batch_size = 32
    num_workers = 8
    window_size = 64
    unit_length = 4

    dl = motion_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        window_size=window_size,
        unit_length=unit_length,
        debug=True
    )
    print(next(iter(dl)).shape)
    print()