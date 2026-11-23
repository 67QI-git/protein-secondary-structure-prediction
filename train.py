import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from loaddata import ADAPTED_FEATURE_SIZE, MAX_LENGTH, PAD_LABEL, FeatureDataset
from model_condgated_cbe_aspp import GatedCBEASPP
from valid import evaluate


def boolean(value):
    value = value.lower()
    if value in ('1', 'true', 'yes'):
        return True
    if value in ('0', 'false', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value.')


def csv_ints(value):
    try:
        return tuple(int(item) for item in value.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError('Expected comma-separated integers.')


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Train and evaluate ESM-RASP on Q8 labels.')
    parser.add_argument('--data_fpath', type=Path, default='/root')
    parser.add_argument('--train_fname', default='train.feat')
    parser.add_argument('--valid_fname', default='valid.feat')
    parser.add_argument('--test_fname', default='test.feat',
                        help='Comma-separated benchmark files. Use an empty string for training only.')
    parser.add_argument('--test_mark', default='cb513', help='Benchmark names, in file order.')
    parser.add_argument('--model_path', type=Path, default='models')
    parser.add_argument('--load_model', type=boolean, nargs='?', const=True, default=False)
    parser.add_argument('--reset_optimizer', type=boolean, nargs='?', const=True, default=True)
    parser.add_argument('--save_acc', type=boolean, nargs='?', const=True, default=True)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out_channel', type=int, default=64)
    parser.add_argument('--lstm_hidden_size', type=int, default=512)
    parser.add_argument('--blstm_layer_num', type=int, default=2)
    parser.add_argument('--bigru_layer_num', type=int, default=3)
    parser.add_argument('--lstm_dropout_rate', type=float, default=0.0)
    parser.add_argument('--output_dropout_rate', type=float, default=0.0)
    parser.add_argument('--window_size_list', type=csv_ints, default=(1, 3, 3, 3))
    parser.add_argument('--dilation_list', type=csv_ints, default=(1, 2, 4, 8))
    parser.add_argument('--ablation', choices=('none', 'bigru', 'bilstm', 'aspp'), default='none')
    parser.add_argument('--model', choices=('condgatedcbeaspp',), default='condgatedcbeaspp')
    parser.add_argument('--label', choices=('ss',), default='ss')
    parser.add_argument('--loss_function', choices=('cel',), default='cel')
    parser.add_argument('--lr_sche', choices=('RlrOP',), default='RlrOP')
    config = parser.parse_args(argv)
    if min(config.batch_size, config.num_epochs, config.out_channel, config.lstm_hidden_size,
           config.blstm_layer_num, config.bigru_layer_num) < 1:
        parser.error('Batch size, epoch count, and recurrent sizes must be positive.')
    if min(config.num_gpus, config.num_workers) < 0:
        parser.error('GPU and worker counts must be nonnegative.')
    config.test_fname_list = [name.strip() for name in config.test_fname.split(',')] if config.test_fname else []
    config.test_mark_list = [name.strip() for name in config.test_mark.split(',')] if config.test_fname_list else []
    if len(config.test_fname_list) != len(config.test_mark_list):
        parser.error('--test_fname and --test_mark must have the same number of entries.')
    if any(not name for name in config.test_fname_list):
        parser.error('--test_fname must contain nonempty filenames.')
    if any(not mark or Path(mark).name != mark or mark == '..' for mark in config.test_mark_list):
        parser.error('--test_mark must contain benchmark names.')
    if len(set(config.test_mark_list)) != len(config.test_mark_list):
        parser.error('--test_mark must contain distinct benchmark names.')
    config.data_max_text_len = MAX_LENGTH
    config.feature_size = ADAPTED_FEATURE_SIZE
    config.class_num = PAD_LABEL + 1
    return config


def get_device(config):
    cuda_available = torch.cuda.is_available() and config.num_gpus > 0
    if config.device == 'cuda' and not cuda_available:
        raise RuntimeError('CUDA requires an available GPU and --num_gpus greater than zero.')
    if config.device == 'cpu' or not cuda_available:
        return torch.device('cpu'), []
    return torch.device('cuda'), list(range(min(config.num_gpus, torch.cuda.device_count())))


def save_checkpoint(state, path):
    with NamedTemporaryFile(dir=path.parent, prefix=path.name, delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        torch.save(state, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = correct = residue_count = 0
    for features, labels in loader:
        features = features.transpose(1, 2).to(device)
        labels = labels.to(device)
        valid = labels.ne(PAD_LABEL)
        optimizer.zero_grad()
        logits = model(features, valid)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        predictions = logits.detach()[:, :PAD_LABEL].argmax(dim=1)
        length = valid.sum().item()
        total_loss += loss.item() * length
        residue_count += length
        correct += (predictions.eq(labels) & valid).sum().item()
    return total_loss / residue_count, correct / residue_count


def main(argv=None):
    config = parse_args(argv)
    device, device_ids = get_device(config)
    print('Using device: {}'.format(device))
    torch.manual_seed(config.seed)
    train_data = FeatureDataset(config.data_fpath / config.train_fname, config)
    valid_data = FeatureDataset(config.data_fpath / config.valid_fname, config)
    test_data = [
        FeatureDataset(config.data_fpath / filename, config)
        for filename in config.test_fname_list
    ]
    if any(dataset.feature_size != train_data.feature_size for dataset in [valid_data] + test_data):
        raise ValueError('Training, validation and benchmark files must use the same feature format.')

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers)
    valid_loader = DataLoader(valid_data, batch_size=1, num_workers=config.num_workers)
    test_loaders = [
        DataLoader(dataset, batch_size=1, num_workers=config.num_workers)
        for dataset in test_data
    ]

    predictor = GatedCBEASPP(out_channel=config.out_channel, config=config).to(device)
    model = nn.DataParallel(predictor, device_ids=device_ids) if len(device_ids) > 1 else predictor
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999),
                                 weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.1, min_lr=1e-9)
    config.model_path.mkdir(parents=True, exist_ok=True)
    last_path = config.model_path / 'Epoch_last.ckpt'
    start_epoch = 0
    best_epoch = 0
    best_q8 = -1.0
    if config.load_model:
        checkpoint = torch.load(last_path, map_location=device)
        if 'feature_size' not in checkpoint:
            raise ValueError('The checkpoint is incompatible with the ESM-RASP model.')
        if checkpoint.get('feature_size') != train_data.feature_size:
            raise ValueError('The checkpoint and training file must use the same feature format.')
        if checkpoint['ablation'] != config.ablation:
            raise ValueError('The checkpoint and run must use the same ablation.')
        predictor.load_state_dict(checkpoint['model'])
        if not config.reset_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_epoch = checkpoint['best_epoch']
        best_q8 = checkpoint['best_q8']
        del checkpoint

    model_directory = config.model_path.resolve()
    log_path = model_directory.with_name(model_directory.name + '.log')
    with log_path.open('a' if config.load_model else 'w', encoding='utf-8') as log:
        for epoch in range(start_epoch, config.num_epochs):
            torch.manual_seed(config.seed + epoch)
            loss, train_q8 = train_epoch(model, train_loader, optimizer, loss_fn, device)
            results = evaluate(model, valid_loader, device)
            scheduler.step(results.q8)
            if results.q8 > best_q8:
                best_epoch, best_q8 = epoch + 1, results.q8

            summary = (
                'Epoch [{}/{}], Loss: {:.5f}, train_Q8: {:.5f}, valid_Q8: {:.5f}, '
                'valid_Q3: {:.5f}, best_epoch: {}'
            ).format(epoch + 1, config.num_epochs, loss, train_q8, results.q8, results.q3, best_epoch)
            print(summary)
            print(summary, file=log, flush=True)
            state = {
                'model': predictor.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'feature_size': train_data.feature_size,
                'ablation': config.ablation,
                'epoch': epoch,
                'best_epoch': best_epoch,
                'best_q8': best_q8,
            }
            save_checkpoint(state, config.model_path / 'Epoch{}.ckpt'.format(epoch + 1))
            save_checkpoint(state, last_path)

        if test_loaders:
            best_path = config.model_path / 'Epoch{}.ckpt'.format(best_epoch)
            predictor.load_state_dict(torch.load(best_path, map_location='cpu')['model'])
            for loader, mark in zip(test_loaders, config.test_mark_list):
                save_path = None
                if config.save_acc:
                    filename = 'acc_{}_Epoch{}_{}.acc'.format(mark, best_epoch, config.model)
                    save_path = config.model_path / filename
                results = evaluate(model, loader, device, save_path)
                summary = ('{} Epoch {}, Q8: {:.5f}, Q3: {:.5f}, '
                           'protein_Q8: {:.5f} ± {:.5f}, edge_Q8: {:.5f}').format(
                    mark, best_epoch, results.q8, results.q3, results.protein_q8,
                    results.protein_q8_std, results.edge_q8
                )
                print(summary)
                print(summary, file=log, flush=True)


if __name__ == '__main__':
    main()
