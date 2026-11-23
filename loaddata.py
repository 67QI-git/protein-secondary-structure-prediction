import random
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


MAX_LENGTH = 700
AA_ALPHABET = 'ACEDGFIHKMLNQPSRTWVYX'
Q8_STATES = 'CBEGIHST'
PAD_LABEL = len(Q8_STATES)
ESM_DIM = 2560
RAW_FEATURE_SIZE = ESM_DIM + len(AA_ALPHABET)
ADAPTED_FEATURE_SIZE = 2 * len(AA_ALPHABET)


class FeatureDataset(Dataset):
    """Read fixed-length chains from a feature file on demand."""

    def __init__(self, path, config):
        self.path = path
        self.length = config.data_max_text_len
        self.offsets = []
        self.feature_size = None

        with open(path, 'rb') as handle:
            for header in iter(handle.readline, b''):
                if not header.strip():
                    continue
                record = len(self.offsets) + 1
                try:
                    rows = int(header)
                except ValueError as exc:
                    raise ValueError('{}: invalid row count in record {}'.format(path, record)) from exc
                if not 1 <= rows <= self.length:
                    raise ValueError('{}: row count in record {} must be between 1 and {}'.format(
                        path, record, self.length
                    ))

                offset = handle.tell()
                width = len(handle.readline().split())
                if width not in (ADAPTED_FEATURE_SIZE, RAW_FEATURE_SIZE):
                    raise ValueError('{}: unsupported feature width {} in record {}'.format(
                        path, width, record
                    ))
                if self.feature_size is None:
                    self.feature_size = width
                elif self.feature_size != width:
                    raise ValueError('{}: mixed feature widths in record {}'.format(path, record))

                for _ in range(2 * rows - 1):
                    if not handle.readline():
                        raise ValueError('{}: incomplete record {}'.format(path, record))
                self.offsets.append((offset, rows))

        if not self.offsets:
            raise ValueError('{}: no feature records'.format(path))

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        offset, rows = self.offsets[index]
        features = np.zeros((self.length, self.feature_size), dtype=np.float32)
        labels = np.full(self.length, PAD_LABEL, dtype=np.int64)

        try:
            with open(self.path, 'rb') as handle, warnings.catch_warnings():
                warnings.simplefilter('error', DeprecationWarning)
                handle.seek(offset)
                for row in features[:rows]:
                    values = np.fromstring(handle.readline(), dtype=np.float32, sep=' ')
                    if values.size != self.feature_size:
                        raise ValueError('expected {} features per residue'.format(self.feature_size))
                    row[:] = values
                for position in range(rows):
                    label = float(handle.readline().strip())
                    if label not in range(PAD_LABEL + 1):
                        raise ValueError('invalid Q8 label {}'.format(label))
                    labels[position] = int(label)
        except (ValueError, DeprecationWarning) as exc:
            raise ValueError('{}: invalid feature or label in record {}: {}'.format(
                self.path, index + 1, exc
            )) from exc

        if not np.isfinite(features[:rows]).all():
            raise ValueError('{}: non-finite features in record {}'.format(self.path, index + 1))

        identity = features[:, -len(AA_ALPHABET):]
        identity_counts = identity.sum(axis=1)
        if not np.all((identity == 0) | (identity == 1)) or (identity_counts > 1).any():
            raise ValueError('{}: invalid one-hot residue identities in record {}'.format(
                self.path, index + 1
            ))
        residue_mask = identity_counts == 1
        residue_count = int(residue_mask.sum())
        if not residue_count or not residue_mask[:residue_count].all():
            raise ValueError('{}: record {} must contain a chain followed by padding'.format(
                self.path, index + 1
            ))

        # In 42-channel records, an identity vector identifies coil entries stored as 8.
        if self.feature_size == ADAPTED_FEATURE_SIZE:
            labels[residue_mask & (labels == PAD_LABEL)] = Q8_STATES.index('C')
        if not np.array_equal(labels != PAD_LABEL, residue_mask):
            raise ValueError('{}: residue identities and padding labels disagree in record {}'.format(
                self.path, index + 1
            ))

        return torch.from_numpy(features), torch.from_numpy(labels)


def read_data(path, config, shuffle=False):
    dataset = FeatureDataset(path, config)
    order = list(range(len(dataset)))
    if shuffle:
        random.shuffle(order)

    features = torch.empty(
        (len(dataset), dataset.length, dataset.feature_size), dtype=torch.float32
    )
    labels = torch.empty((len(dataset), dataset.length), dtype=torch.long)
    for row, index in enumerate(order):
        features[row], labels[row] = dataset[index]
    return features, labels
