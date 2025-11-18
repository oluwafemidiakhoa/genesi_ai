"""
Unit tests for data loading and preprocessing.
"""

import pytest
import torch
from genesis_rna.tokenization import RNATokenizer
from genesis_rna.data import (
    RNAPretrainSample,
    RNAPretrainDataset,
    RNASequenceDataset,
    collate_pretrain_batch,
    create_dummy_dataset,
)


class TestDataStructures:
    """Test data structures and dataset classes"""

    @pytest.fixture
    def tokenizer(self):
        """Create tokenizer"""
        return RNATokenizer()

    @pytest.fixture
    def sample_sequences(self):
        """Create sample RNA sequences"""
        return [
            "ACGUACGUACGU",
            "GGCCGGCCGGCC",
            "UUAAUUAAUUAA",
        ]

    def test_rna_pretrain_sample_creation(self):
        """Test creating RNAPretrainSample"""
        sample = RNAPretrainSample(
            seq="ACGUACGU",
            struct_labels=[0, 1, 1, 2, 2, 1, 1, 0],
            pair_indices=[(1, 6), (2, 5)],
        )

        assert sample.seq == "ACGUACGU"
        assert len(sample.struct_labels) == len(sample.seq)
        assert len(sample.pair_indices) == 2

    def test_rna_pretrain_sample_validation(self):
        """Test sample validation"""
        # This should raise an error (length mismatch)
        with pytest.raises(AssertionError):
            RNAPretrainSample(
                seq="ACGU",
                struct_labels=[0, 1, 1],  # Wrong length
            )

    def test_rna_pretrain_dataset_creation(self, tokenizer):
        """Test creating RNAPretrainDataset"""
        samples = [
            RNAPretrainSample(seq="ACGUACGU"),
            RNAPretrainSample(seq="GGCCGGCC"),
        ]

        dataset = RNAPretrainDataset(
            samples=samples,
            tokenizer=tokenizer,
            max_len=32,
        )

        assert len(dataset) == 2

    def test_rna_pretrain_dataset_getitem(self, tokenizer):
        """Test getting item from dataset"""
        samples = [
            RNAPretrainSample(
                seq="ACGUACGU",
                struct_labels=[0, 1, 1, 2, 2, 1, 1, 0],
            )
        ]

        dataset = RNAPretrainDataset(
            samples=samples,
            tokenizer=tokenizer,
            max_len=32,
        )

        item = dataset[0]

        # Check all required keys exist
        assert 'input_ids' in item
        assert 'mlm_labels' in item
        assert 'struct_labels' in item
        assert 'pair_matrix' in item
        assert 'attention_mask' in item

        # Check shapes
        assert item['input_ids'].shape == (32,)
        assert item['mlm_labels'].shape == (32,)
        assert item['struct_labels'].shape == (32,)
        assert item['pair_matrix'].shape == (32, 32)
        assert item['attention_mask'].shape == (32,)

    def test_rna_sequence_dataset(self, tokenizer, sample_sequences):
        """Test simple sequence dataset"""
        dataset = RNASequenceDataset(
            sequences=sample_sequences,
            tokenizer=tokenizer,
            max_len=32,
        )

        assert len(dataset) == len(sample_sequences)

        item = dataset[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert item['input_ids'].shape == (32,)

    def test_collate_pretrain_batch(self, tokenizer):
        """Test batch collation"""
        samples = [
            RNAPretrainSample(seq="ACGU"),
            RNAPretrainSample(seq="GGCC"),
        ]

        dataset = RNAPretrainDataset(samples, tokenizer, max_len=16)

        batch_list = [dataset[0], dataset[1]]
        batch = collate_pretrain_batch(batch_list)

        # Check batch dimensions
        assert batch['input_ids'].shape == (2, 16)
        assert batch['mlm_labels'].shape == (2, 16)
        assert batch['attention_mask'].shape == (2, 16)
        assert batch['pair_matrix'].shape == (2, 16, 16)

    def test_create_dummy_dataset(self):
        """Test creating dummy dataset"""
        samples = create_dummy_dataset(
            num_samples=10,
            min_len=50,
            max_len=100,
            with_structure=True,
        )

        assert len(samples) == 10

        for sample in samples:
            assert isinstance(sample, RNAPretrainSample)
            assert 50 <= len(sample.seq) <= 100
            assert sample.struct_labels is not None
            assert sample.pair_indices is not None

    def test_dummy_dataset_without_structure(self):
        """Test dummy dataset without structure annotations"""
        samples = create_dummy_dataset(
            num_samples=5,
            with_structure=False,
        )

        assert len(samples) == 5

        for sample in samples:
            assert sample.struct_labels is None
            assert sample.pair_indices is None

    def test_dataset_masking_statistics(self, tokenizer):
        """Test MLM masking produces expected statistics"""
        samples = [RNAPretrainSample(seq="ACGU" * 20)]

        dataset = RNAPretrainDataset(
            samples=samples,
            tokenizer=tokenizer,
            max_len=128,
            mlm_probability=0.15,
        )

        # Get multiple samples to check masking randomness
        masked_counts = []
        for _ in range(100):
            item = dataset[0]
            num_masked = (item['mlm_labels'] != -100).sum().item()
            masked_counts.append(num_masked)

        avg_masked = sum(masked_counts) / len(masked_counts)

        # Should be roughly 15% of non-special tokens
        # Rough check (within 5-25% range)
        assert 5 < avg_masked < 25

    def test_dataset_with_dataloader(self, tokenizer):
        """Test dataset with PyTorch DataLoader"""
        from torch.utils.data import DataLoader

        samples = create_dummy_dataset(num_samples=20)
        dataset = RNAPretrainDataset(samples, tokenizer, max_len=64)

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collate_pretrain_batch,
        )

        # Test one batch
        batch = next(iter(dataloader))

        assert batch['input_ids'].shape == (4, 64)
        assert batch['attention_mask'].shape == (4, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
