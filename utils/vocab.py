import torch
from torch.nn import functional

class Vocab:
    def __init__(self, paths):
        self.chars = None
        self.image_paths = []
        self.labels = []
        self.load_dataset(paths)

    def load_dataset(self, paths):
        if isinstance(paths, str):
            paths = [paths]

        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) != 2:
                        continue
                    img_path, label = parts
                    self.image_paths.append(img_path)
                    self.labels.append(label)
        print(f"Total {len(self.image_paths)} images loaded from {len(paths)} file(s)")

    def create_vocab(self):
        letters = "".join(self.labels)
        unique_chars = sorted(set(letters))

        self.chars = "".join(unique_chars)

        self.char_2_idx = {char: idx + 2 for idx, char in enumerate(self.chars)}
        self.blank_index = 1  # index cho CTC blank
        self.idx_2_char = {idx: char for char, idx in self.char_2_idx.items()}

        print(f"Vocab size (with blank): {len(self.chars) + 1}")
        print(f"char_2_idx: {self.char_2_idx}")

    def encode(self, input_sequence):
        max_label_len = max(len(label) for label in self.labels)
        encoded = torch.tensor(
            [self.char_2_idx.get(char, 0) for char in input_sequence], dtype=torch.long  #  -> 0 (pad)
        )
        label_len = len(encoded)
        lengths = torch.tensor(label_len, dtype=torch.long)
        padded = functional.pad(encoded, (0, max_label_len - label_len))
        return padded, lengths

    def decode(self, encode_sequences):
        decode_sequences = []

        for seq in encode_sequences:
            decode_label = []
            prev_token = None
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()

            for token in seq:
                if token != self.blank_index and token != 0 and token != prev_token:
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
            chars = [self.idx_2_char.get(idx, '') for idx in label if idx != 0]
            idx_2_labels.append(''.join(chars))
        return idx_2_labels

if __name__ == 'main':
    vocab = Vocab([
    "/content/drive/MyDrive/scene-text-ocr/train.txt",
    "/content/drive/MyDrive/scene-text-ocr/val.txt",
    "/content/drive/MyDrive/scene-text-ocr/test.txt"
    ])
    vocab.create_vocab()
