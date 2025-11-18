"""
Unit tests for RNA tokenization.
"""

import pytest
import torch
from genesis_rna.tokenization import RNATokenizer, RNATokenizerConfig


class TestRNATokenizer:
    """Test suite for RNATokenizer"""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance"""
        return RNATokenizer()

    def test_tokenizer_initialization(self, tokenizer):
        """Test tokenizer is initialized correctly"""
        assert tokenizer.vocab_size == 9  # 4 special + 5 nucleotides
        assert tokenizer.pad_id == 0
        assert tokenizer.mask_id == 1
        assert tokenizer.cls_id == 2
        assert tokenizer.sep_id == 3

    def test_encode_simple_sequence(self, tokenizer):
        """Test encoding a simple RNA sequence"""
        seq = "ACGU"
        max_len = 10
        encoded = tokenizer.encode(seq, max_len)

        assert encoded.shape == (max_len,)
        assert encoded[0] == tokenizer.cls_id  # [CLS]
        assert encoded[-1] == tokenizer.pad_id  # Padding
        # Check nucleotides are encoded
        assert all(encoded[i] != tokenizer.pad_id for i in range(1, len(seq) + 1))

    def test_encode_with_padding(self, tokenizer):
        """Test encoding handles padding correctly"""
        seq = "ACG"
        max_len = 10
        encoded = tokenizer.encode(seq, max_len)

        # Count padding tokens
        num_pad = (encoded == tokenizer.pad_id).sum().item()
        assert num_pad == max_len - len(seq) - 2  # -2 for [CLS] and [SEP]

    def test_encode_with_truncation(self, tokenizer):
        """Test encoding handles truncation"""
        seq = "ACGU" * 100  # Long sequence
        max_len = 10
        encoded = tokenizer.encode(seq, max_len)

        assert encoded.shape == (max_len,)
        assert len(encoded) == max_len

    def test_decode(self, tokenizer):
        """Test decoding token IDs back to sequence"""
        original_seq = "ACGUACGU"
        max_len = 20
        encoded = tokenizer.encode(original_seq, max_len)
        decoded = tokenizer.decode(encoded)

        # Remove [CLS], [SEP], and padding
        assert original_seq in decoded or decoded in original_seq

    def test_random_mask(self, tokenizer):
        """Test random masking for MLM"""
        seq = "ACGU" * 10
        max_len = 50
        input_ids = tokenizer.encode(seq, max_len)

        masked_ids, labels = tokenizer.random_mask(input_ids, mlm_prob=0.15)

        # Check shapes match
        assert masked_ids.shape == input_ids.shape
        assert labels.shape == input_ids.shape

        # Check some tokens are masked
        num_masked = (masked_ids == tokenizer.mask_id).sum().item()
        assert num_masked > 0

        # Check labels have -100 for non-masked positions
        num_ignore = (labels == -100).sum().item()
        assert num_ignore > 0

    def test_batch_encode(self, tokenizer):
        """Test batch encoding"""
        sequences = ["ACGU", "GGCC", "UUAA"]
        max_len = 10
        batch = tokenizer.batch_encode(sequences, max_len)

        assert batch.shape == (len(sequences), max_len)
        assert batch.dtype == torch.long

    def test_unknown_nucleotide(self, tokenizer):
        """Test handling of unknown nucleotides"""
        seq = "ACGX"  # X is unknown
        max_len = 10
        encoded = tokenizer.encode(seq, max_len)

        # Should encode X as N
        assert encoded[4] == tokenizer.token_to_id["N"]

    def test_lowercase_handling(self, tokenizer):
        """Test lowercase sequences are handled correctly"""
        seq_upper = "ACGU"
        seq_lower = "acgu"
        max_len = 10

        encoded_upper = tokenizer.encode(seq_upper, max_len)
        encoded_lower = tokenizer.encode(seq_lower, max_len)

        # Should be identical
        assert torch.equal(encoded_upper, encoded_lower)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
