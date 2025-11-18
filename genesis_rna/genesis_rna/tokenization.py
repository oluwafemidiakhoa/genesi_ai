"""
RNA Tokenization Module

Provides tokenization for RNA sequences with support for:
- Nucleotide encoding (A, C, G, U, N)
- Special tokens ([PAD], [MASK], [CLS], [SEP])
- Structure labels (STEM, LOOP, BULGE, HAIRPIN)
- Random masking for MLM pretraining
"""

import torch
from dataclasses import dataclass
from typing import List, Tuple

# RNA vocabulary
NUC_VOCAB = ["A", "C", "G", "U", "N"]
SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[CLS]", "[SEP]"]
STRUCT_LABELS = ["NONE", "STEM", "LOOP", "BULGE", "HAIRPIN"]


@dataclass
class RNATokenizerConfig:
    """Configuration for RNA tokenizer"""
    add_structure_tokens: bool = False
    mlm_probability: float = 0.15
    mask_token_probability: float = 0.8  # Prob of replacing with [MASK]
    random_token_probability: float = 0.1  # Prob of replacing with random token
    # remaining probability: keep original token


class RNATokenizer:
    """
    Tokenizer for RNA sequences.

    Converts RNA sequences to token IDs and provides masking utilities
    for masked language modeling (MLM) pretraining.

    Example:
        >>> tokenizer = RNATokenizer()
        >>> seq = "ACGUACGU"
        >>> tokens = tokenizer.encode(seq, max_len=16)
        >>> masked_tokens, labels = tokenizer.random_mask(tokens)
    """

    def __init__(self, cfg: RNATokenizerConfig = None):
        self.cfg = cfg or RNATokenizerConfig()

        # Build vocabulary
        self.vocab = SPECIAL_TOKENS + NUC_VOCAB
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

        # Special token IDs
        self.pad_id = self.token_to_id["[PAD]"]
        self.mask_id = self.token_to_id["[MASK]"]
        self.cls_id = self.token_to_id["[CLS]"]
        self.sep_id = self.token_to_id["[SEP]"]

        # Nucleotide IDs for random replacement
        self.nucleotide_ids = [self.token_to_id[nuc] for nuc in NUC_VOCAB[:4]]  # A, C, G, U

        self.vocab_size = len(self.vocab)

    def encode(self, seq: str, max_len: int) -> torch.Tensor:
        """
        Encode an RNA sequence to token IDs.

        Args:
            seq: RNA sequence string (e.g., "ACGUACGU")
            max_len: Maximum sequence length (will pad or truncate)

        Returns:
            Token IDs tensor of shape [max_len]
        """
        tokens = [self.cls_id]

        # Convert sequence to tokens
        for ch in seq.upper():
            if ch in self.token_to_id:
                tokens.append(self.token_to_id[ch])
            else:
                # Unknown nucleotide -> N
                tokens.append(self.token_to_id["N"])

        tokens.append(self.sep_id)

        # Truncate or pad
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens += [self.pad_id] * (max_len - len(tokens))

        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs back to RNA sequence.

        Args:
            token_ids: Token IDs tensor

        Returns:
            RNA sequence string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = []
        for idx in token_ids:
            token = self.id_to_token.get(idx, "N")
            if token not in SPECIAL_TOKENS:
                tokens.append(token)

        return "".join(tokens)

    def random_mask(
        self,
        input_ids: torch.Tensor,
        mlm_prob: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random masking for MLM pretraining.

        Strategy (following BERT):
        - Select mlm_prob (default 15%) of tokens
        - Of selected tokens:
            - 80% -> replace with [MASK]
            - 10% -> replace with random token
            - 10% -> keep original

        Args:
            input_ids: Token IDs tensor [B, L] or [L]
            mlm_prob: Probability of masking each token (default: from config)

        Returns:
            Tuple of (masked_input_ids, labels)
            - masked_input_ids: Input with masked tokens
            - labels: Original tokens for masked positions, -100 elsewhere
        """
        if mlm_prob is None:
            mlm_prob = self.cfg.mlm_probability

        labels = input_ids.clone()

        # Create mask for tokens to be masked (excluding special tokens)
        probability_matrix = torch.full(input_ids.shape, mlm_prob)

        # Don't mask special tokens
        special_tokens_mask = (
            (input_ids == self.pad_id) |
            (input_ids == self.cls_id) |
            (input_ids == self.sep_id) |
            (input_ids == self.mask_id)
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Sample which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels to -100 for non-masked tokens (ignore in loss)
        labels[~masked_indices] = -100

        # Create masked input
        input_ids_masked = input_ids.clone()

        # 80% of the time: replace with [MASK]
        mask_token_mask = (
            torch.bernoulli(torch.full(input_ids.shape, self.cfg.mask_token_probability)).bool()
            & masked_indices
        )
        input_ids_masked[mask_token_mask] = self.mask_id

        # 10% of the time: replace with random nucleotide
        random_token_mask = (
            torch.bernoulli(
                torch.full(input_ids.shape, self.cfg.random_token_probability / (1 - self.cfg.mask_token_probability))
            ).bool()
            & masked_indices
            & ~mask_token_mask
        )
        random_tokens = torch.randint(
            len(self.nucleotide_ids),
            input_ids.shape,
            dtype=torch.long
        )
        random_token_ids = torch.tensor(self.nucleotide_ids)[random_tokens]
        input_ids_masked[random_token_mask] = random_token_ids[random_token_mask]

        # Remaining 10%: keep original token (already in input_ids_masked)

        return input_ids_masked, labels

    def batch_encode(self, sequences: List[str], max_len: int) -> torch.Tensor:
        """
        Encode a batch of RNA sequences.

        Args:
            sequences: List of RNA sequence strings
            max_len: Maximum sequence length

        Returns:
            Tensor of shape [batch_size, max_len]
        """
        return torch.stack([self.encode(seq, max_len) for seq in sequences])

    def __len__(self) -> int:
        """Return vocabulary size"""
        return self.vocab_size

    def __repr__(self) -> str:
        return f"RNATokenizer(vocab_size={self.vocab_size}, mlm_prob={self.cfg.mlm_probability})"
