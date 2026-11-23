from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

from loaddata import AA_ALPHABET, PAD_LABEL, Q8_STATES


Evaluation = namedtuple(
    'Evaluation', 'q8 q3 edge_q8 protein_q8 protein_q8_std protein_edge_q8 protein_edge_q8_std'
)


def edge_mask(labels, valid):
    edges = torch.zeros_like(valid)
    edges[:, 0] = True
    edges[:, -1] = True
    transitions = (labels[:, 1:] != labels[:, :-1]) | (valid[:, 1:] != valid[:, :-1])
    edges[:, :-1] |= transitions
    edges[:, 1:] |= transitions
    return edges & valid


def acc(pred_list, label_list, ignore_cls=PAD_LABEL):
    valid = label_list.ne(ignore_cls)
    edges = edge_mask(label_list, valid)
    hits = pred_list.eq(label_list)
    return ((hits & valid).sum().item() / max(valid.sum().item(), 1),
            (hits & edges).sum().item() / max(edges.sum().item(), 1))


@torch.no_grad()
def evaluate(model, loader, device, save_path=None, channel_first=False):
    model.eval()
    q3_classes = dict(C=0, S=0, T=0, E=1, B=1, H=2, G=2, I=2)
    q3_map = torch.tensor([q3_classes[state] for state in Q8_STATES] + [3], device=device)
    protein_q8 = []
    protein_edge_q8 = []
    records = []
    residue_count = q8_correct = q3_correct = edge_count = edge_correct = 0

    for features, labels in loader:
        features = features if channel_first else features.transpose(1, 2)
        labels = labels.to(device)
        valid = labels.ne(PAD_LABEL)
        logits = model(features.to(device), valid)
        predictions = logits[:, :len(Q8_STATES)].argmax(dim=1)
        hits = predictions.eq(labels)
        edges = edge_mask(labels, valid)
        counts = torch.stack((
            valid.sum(dim=1),
            (hits & valid).sum(dim=1),
            (q3_map[predictions].eq(q3_map[labels]) & valid).sum(dim=1),
            edges.sum(dim=1),
            (hits & edges).sum(dim=1),
        ), dim=1).tolist()
        if save_path is not None:
            residues = features[:, -len(AA_ALPHABET):, :]
            identity = residues.argmax(dim=1).tolist()
            present = residues.ne(0).any(dim=1).tolist()

        for i, (length, correct, correct_q3, n_edges, correct_edges) in enumerate(counts):
            if length == 0:
                continue
            residue_count += length
            q8_correct += correct
            q3_correct += correct_q3
            edge_count += n_edges
            edge_correct += correct_edges
            score = correct / length
            protein_q8.append(score)
            protein_edge_q8.append(correct_edges / n_edges)
            if save_path is not None:
                seq = ''.join(AA_ALPHABET[index] for index, is_present in
                              zip(identity[i], present[i]) if is_present)
                records.append((seq, score))

    if residue_count == 0:
        raise ValueError('Evaluation requires at least one labeled residue.')
    if save_path is not None:
        with Path(save_path).open('w', encoding='utf-8') as output:
            for sequence, score in records:
                output.write('{},{}\n'.format(sequence, score))

    return Evaluation(
        q8_correct / residue_count,
        q3_correct / residue_count,
        edge_correct / edge_count,
        np.mean(protein_q8),
        np.std(protein_q8),
        np.mean(protein_edge_q8),
        np.std(protein_edge_q8),
    )


def get_valid_acc(config, model, valid_loader, epoch, test_mark, save_acc=True):
    save_path = None
    if save_acc:
        filename = 'acc_{}_Epoch{}_{}.acc'.format(test_mark, epoch + 1, config.model)
        save_path = Path(config.save_acc_fpath) / filename
    results = evaluate(
        model, valid_loader, config.device, save_path, channel_first=True
    )
    return (results.protein_q8, results.protein_edge_q8, results.protein_q8_std,
            results.protein_edge_q8_std)
