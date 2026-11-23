import argparse
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from loaddata import AA_ALPHABET, ESM_DIM, MAX_LENGTH, Q8_STATES


AA_INDEX = {residue: index for index, residue in enumerate(AA_ALPHABET)}
Q8_INDEX = {state: index for index, state in enumerate(Q8_STATES)}
COIL_ALIASES = {' ': 'C', '-': 'C', 'L': 'C', 'P': 'C'}
ESM_LAYER = 36


def normalize_sequence(sequence):
    uppercase_residues = (residue.upper() for residue in sequence)
    return ''.join(residue if residue in AA_INDEX else 'X' for residue in uppercase_residues)


def encode_one_hot(sequence):
    indices = [AA_INDEX.get(residue, AA_INDEX['X']) for residue in sequence]
    return np.eye(len(AA_ALPHABET), dtype=np.float32)[indices]


def read_ss(path):
    with open(path, encoding='utf-8-sig') as source:
        for line_number, line in enumerate(source, 1):
            line = line.rstrip('\r\n')
            if not line.strip():
                continue
            sequence, comma, states = line.partition(',')
            if not comma:
                raise ValueError('{}:{}: expected a sequence and comma-separated DSSP states'.format(
                    path, line_number
                ))
            yield sequence.strip(), states, '{}:{}'.format(path, line_number)


def read_dssp(path, executable):
    result = subprocess.run(
        [executable, '--output-format=dssp', str(path)],
        check=True, stdout=subprocess.PIPE, universal_newlines=True
    )

    chains = {}
    in_residues = False
    for line in result.stdout.splitlines():
        if not in_residues:
            in_residues = line.split()[:2] == ['#', 'RESIDUE']
            continue
        if not line.strip():
            continue
        if len(line) < 17:
            raise ValueError('{}: invalid DSSP residue record'.format(path))
        if line[9] == ' ' or line[13] == '!':
            continue
        residues, states = chains.setdefault(line[11], ([], []))
        residue = line[13]
        # DSSP denotes disulfide-bonded cysteines with lowercase bridge IDs.
        residues.append('C' if residue.islower() else residue)
        states.append(line[16])

    if not chains:
        raise ValueError('{}: no DSSP-annotated chains'.format(path))
    for chain, (residues, states) in chains.items():
        yield ''.join(residues), ''.join(states), '{} chain {!r}'.format(path, chain)


def prepare_chains(records):
    seen = set()
    for sequence, states, identifier in records:
        sequence = normalize_sequence(sequence)
        if not sequence or len(sequence) != len(states):
            raise ValueError('{}: sequence and DSSP states must have the same nonzero length'.format(
                identifier
            ))
        if len(sequence) > MAX_LENGTH:
            raise ValueError('{}: chain length {} exceeds {}'.format(
                identifier, len(sequence), MAX_LENGTH
            ))

        try:
            labels = [Q8_INDEX[COIL_ALIASES.get(state, state)] for state in states.upper()]
        except KeyError as exc:
            raise ValueError('{}: invalid DSSP state {!r}'.format(identifier, exc.args[0])) from exc

        if sequence not in seen:
            seen.add(sequence)
            yield sequence, labels


def write_features(chains, output, model, batch_converter, device):
    temp_path = None
    try:
        with NamedTemporaryFile(
            mode='w', encoding='ascii', newline='\n', dir=str(output.parent),
            prefix=output.name + '.', suffix='.tmp', delete=False
        ) as handle:
            temp_path = Path(handle.name)
            with torch.no_grad():
                for sequence, labels in chains:
                    _, _, tokens = batch_converter([('sequence', sequence)])
                    result = model(tokens.to(device), repr_layers=[ESM_LAYER], return_contacts=False)
                    embeddings = result['representations'][ESM_LAYER][0, 1:len(sequence) + 1]
                    if embeddings.shape != (len(sequence), ESM_DIM):
                        raise ValueError('unexpected ESM residue embedding shape {}'.format(
                            tuple(embeddings.shape)
                        ))

                    features = np.concatenate((embeddings.float().cpu().numpy(),
                                               encode_one_hot(sequence)), axis=1)
                    handle.write('{}\n'.format(len(sequence)))
                    np.savetxt(handle, features, fmt='%.9g')
                    handle.writelines('{}\n'.format(label) for label in labels)

        os.replace(str(temp_path), str(output))
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description='Cache frozen ESM-2 embeddings for Q8 training.')
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--input', type=Path, default=Path('/root/sequence.ss'),
                        help='Input .ss file with sequence and DSSP labels')
    inputs.add_argument('--pdb', nargs='+', type=Path,
                        help='PDB coordinate files to annotate with mkdssp')
    parser.add_argument('--dssp-executable', default='mkdssp', help='Path to mkdssp 4 or later')
    parser.add_argument('--output', type=Path, default=Path('/root/sequence.feat'))
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda or cuda:N')
    args = parser.parse_args()

    input_paths = args.pdb or [args.input]
    if args.output.resolve() in {path.resolve() for path in input_paths}:
        parser.error('input and output must be different files')

    if args.pdb:
        records = (record for path in args.pdb for record in read_dssp(path, args.dssp_executable))
    else:
        records = read_ss(args.input)
    chains = list(prepare_chains(records))
    if not chains:
        parser.error('input contains no protein chains')

    import esm

    device_name = args.device
    if device_name == 'auto':
        device_name = 'cpu'
        if (torch.cuda.is_available()
                and torch.cuda.get_device_properties(0).total_memory >= 15 * 1024 ** 3):
            device_name = 'cuda'
    device = torch.device(device_name)
    print('Using device: {}'.format(device))
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model = model.to(device).eval().requires_grad_(False)
    write_features(chains, args.output, model, alphabet.get_batch_converter(), device)


if __name__ == '__main__':
    main()
