"""
Data Loading Module for Genesis RNA

Provides dataset classes for RNA pretraining with support for:
- Masked language modeling (MLM)
- Secondary structure prediction
- Base-pair prediction
- Mutation effect prediction
"""

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class RNAPretrainSample:
    """
    Single RNA sequence sample for pretraining.

    Attributes:
        seq: RNA sequence string (e.g., "ACGUACGU...")
        struct_labels: Optional list of structure labels per position
                      (indices into STRUCT_LABELS: NONE, STEM, LOOP, BULGE, HAIRPIN)
        pair_indices: Optional list of (i, j) tuples indicating base-pair positions
        metadata: Optional dictionary with additional info (e.g., source, organism)
    """
    seq: str
    struct_labels: Optional[List[int]] = None
    pair_indices: Optional[List[Tuple[int, int]]] = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        """Validate sample data"""
        if self.struct_labels is not None:
            assert len(self.struct_labels) == len(self.seq), \
                "struct_labels length must match sequence length"


class RNAPretrainDataset(Dataset):
    """
    Dataset for RNA pretraining with multi-task learning.

    Supports:
    1. Masked Language Modeling (MLM)
    2. Secondary structure prediction
    3. Base-pair prediction

    Example:
        >>> samples = [
        ...     RNAPretrainSample(seq="ACGUACGU", struct_labels=[0,1,1,2,2,1,1,0]),
        ...     RNAPretrainSample(seq="GGCCAAUU")
        ... ]
        >>> dataset = RNAPretrainDataset(samples, tokenizer, max_len=64)
        >>> batch = dataset[0]
    """

    def __init__(
        self,
        samples: List[RNAPretrainSample],
        tokenizer,
        max_len: int = 512,
        mlm_probability: float = 0.15,
    ):
        """
        Args:
            samples: List of RNAPretrainSample objects
            tokenizer: RNATokenizer instance
            max_len: Maximum sequence length (will pad/truncate)
            mlm_probability: Probability of masking each token
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mlm_probability = mlm_probability

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.

        Returns:
            Dictionary with keys:
                - input_ids: Masked token IDs [max_len]
                - mlm_labels: Original tokens for masked positions [max_len]
                - struct_labels: Structure type labels [max_len]
                - pair_matrix: Base-pair adjacency matrix [max_len, max_len]
                - attention_mask: Binary mask for valid positions [max_len]
        """
        sample = self.samples[idx]

        # Encode sequence
        input_ids = self.tokenizer.encode(sample.seq, self.max_len)

        # Apply random masking for MLM
        input_ids_masked, mlm_labels = self.tokenizer.random_mask(
            input_ids,
            mlm_prob=self.mlm_probability
        )

        # Process structure labels
        if sample.struct_labels is not None:
            # Pad/truncate structure labels to match max_len
            # Account for [CLS] and [SEP] tokens
            struct_labels = [0] + sample.struct_labels + [0]  # 0 = NONE for special tokens
            if len(struct_labels) > self.max_len:
                struct_labels = struct_labels[:self.max_len]
            else:
                struct_labels += [0] * (self.max_len - len(struct_labels))
            struct_labels = torch.tensor(struct_labels, dtype=torch.long)
        else:
            # No structure labels available → ignore in loss
            struct_labels = torch.full((self.max_len,), -100, dtype=torch.long)

        # Process base-pair indices
        pair_matrix = torch.zeros((self.max_len, self.max_len), dtype=torch.float32)
        if sample.pair_indices:
            for i, j in sample.pair_indices:
                # Offset by 1 for [CLS] token
                i_offset = i + 1
                j_offset = j + 1
                if i_offset < self.max_len and j_offset < self.max_len:
                    pair_matrix[i_offset, j_offset] = 1.0
                    pair_matrix[j_offset, i_offset] = 1.0  # Symmetric

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.pad_id).long()

        return {
            "input_ids": input_ids_masked,
            "mlm_labels": mlm_labels,
            "struct_labels": struct_labels,
            "pair_matrix": pair_matrix,
            "attention_mask": attention_mask,
        }


class RNASequenceDataset(Dataset):
    """
    Simple dataset for raw RNA sequences (for inference or fine-tuning).

    Example:
        >>> sequences = ["ACGUACGU", "GGCCAAUU", "UUAACCGG"]
        >>> dataset = RNASequenceDataset(sequences, tokenizer, max_len=64)
    """

    def __init__(
        self,
        sequences: List[str],
        tokenizer,
        max_len: int = 512,
    ):
        """
        Args:
            sequences: List of RNA sequence strings
            tokenizer: RNATokenizer instance
            max_len: Maximum sequence length
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get encoded sequence"""
        seq = self.sequences[idx]
        input_ids = self.tokenizer.encode(seq, self.max_len)
        attention_mask = (input_ids != self.tokenizer.pad_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def collate_pretrain_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for pretraining batches.

    Stacks individual samples into batched tensors.

    Args:
        batch: List of sample dictionaries from RNAPretrainDataset

    Returns:
        Batched dictionary with shape [batch_size, ...]
    """
    if len(batch) == 0:
        return {}

    # Get all keys from first sample
    keys = batch[0].keys()

    # Stack each field
    collated = {}
    for key in keys:
        collated[key] = torch.stack([sample[key] for sample in batch], dim=0)

    return collated


def create_dummy_dataset(
    num_samples: int = 100,
    min_len: int = 50,
    max_len: int = 200,
    with_structure: bool = False,
) -> List[RNAPretrainSample]:
    """
    Create a dummy dataset for testing.

    Args:
        num_samples: Number of samples to generate
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        with_structure: Whether to include dummy structure labels

    Returns:
        List of RNAPretrainSample objects
    """
    samples = []
    nucleotides = ['A', 'C', 'G', 'U']

    for _ in range(num_samples):
        # Random sequence length
        seq_len = np.random.randint(min_len, max_len + 1)

        # Random sequence
        seq = ''.join(np.random.choice(nucleotides, size=seq_len))

        # Optional structure labels
        struct_labels = None
        pair_indices = None

        if with_structure:
            # Random structure labels (0-4 for NONE, STEM, LOOP, BULGE, HAIRPIN)
            struct_labels = np.random.randint(0, 5, size=seq_len).tolist()

            # Random base pairs (simplified, not necessarily valid)
            num_pairs = seq_len // 10
            pair_indices = []
            for _ in range(num_pairs):
                i = np.random.randint(0, seq_len)
                j = np.random.randint(0, seq_len)
                if i != j:
                    pair_indices.append((min(i, j), max(i, j)))

        samples.append(RNAPretrainSample(
            seq=seq,
            struct_labels=struct_labels,
            pair_indices=pair_indices,
        ))

    return samples


# Utility functions for loading real data (placeholders for now)

def load_rnacentral_data(file_path: str, max_samples: Optional[int] = None) -> List[RNAPretrainSample]:
    """
    Load RNA sequences from RNAcentral format.

    TODO: Implement actual parsing logic for RNAcentral files.

    Args:
        file_path: Path to RNAcentral data file
        max_samples: Maximum number of samples to load

    Returns:
        List of RNAPretrainSample objects
    """
    raise NotImplementedError("RNAcentral data loading not yet implemented")


def load_rfam_data(file_path: str, max_samples: Optional[int] = None) -> List[RNAPretrainSample]:
    """
    Load RNA sequences from Rfam format with structure annotations.

    TODO: Implement actual parsing logic for Rfam files.

    Args:
        file_path: Path to Rfam data file
        max_samples: Maximum number of samples to load

    Returns:
        List of RNAPretrainSample objects
    """
    raise NotImplementedError("Rfam data loading not yet implemented")
