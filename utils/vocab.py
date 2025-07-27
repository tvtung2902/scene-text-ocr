import yaml
import torch
from torch.nn import functional

class Vocab:
    def __init__(self, vocab_path):
        self.chars = None
        self.char_2_idx = {}
        self.idx_2_char = {}
        self.pad_index = 0
        self.blank_index = 1
        self.vocab_path = vocab_path
        self.create_vocab()

    def create_vocab(self):
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            vocab_yaml = yaml.safe_load(f)
            self.chars = vocab_yaml['vocab']

        self.char_2_idx = {char: idx + 2 for idx, char in enumerate(self.chars)}
        self.char_2_idx["<pad>"] = self.pad_index
        self.char_2_idx["<blank>"] = self.blank_index

        self.idx_2_char = {idx: char for char, idx in self.char_2_idx.items()}

        print(f"Vocab size (with <pad> and <blank>): {len(self.char_2_idx)}")
        print(f"char_2_idx: {self.char_2_idx}")

    def encode(self, input_sequence, max_len=None):
        encoded = torch.tensor(
            [self.char_2_idx.get(char, self.pad_index) for char in input_sequence],
            dtype=torch.long
        )
        label_len = len(encoded)
        if max_len is None:
            max_len = label_len
        lengths = torch.tensor(label_len, dtype=torch.long)
        padded = functional.pad(encoded, (0, max_len - label_len))
        return padded, lengths

    def decode(self, encode_sequences):
        decode_sequences = []
        for seq in encode_sequences:
            decode_label = []
            prev_token = None
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            for token in seq:
                if token != self.blank_index and token != self.pad_index and token != prev_token:
                    char = self.idx_2_char.get(token, '')
                    decode_label.append(char)
                prev_token = token
            decode_sequences.append(''.join(decode_label))
        return decode_sequences

    def idx_2_labels(self, labels):
        idx_2_labels = []
        for label in labels:
            if isinstance(label, torch.Tensor):
                label = label.tolist()
            chars = [self.idx_2_char.get(idx, '') for idx in label if idx != self.pad_index]
            idx_2_labels.append(''.join(chars))
        return idx_2_labels
