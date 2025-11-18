"""
Unit tests for model forward passes and architecture.
"""

import pytest
import torch
from genesis_rna.config import GenesisRNAConfig, GenesisRNAConfigSmall
from genesis_rna.model import (
    RNAEmbedding,
    TransformerBlock,
    GenesisRNAEncoder,
    GenesisRNAModel,
)


class TestModelArchitecture:
    """Test suite for model architecture"""

    @pytest.fixture
    def config(self):
        """Create a small config for fast testing"""
        return GenesisRNAConfigSmall()

    @pytest.fixture
    def batch(self):
        """Create a dummy batch"""
        return {
            'input_ids': torch.randint(0, 9, (4, 64)),  # batch=4, seq_len=64
            'attention_mask': torch.ones(4, 64, dtype=torch.long),
        }

    def test_rna_embedding(self, config):
        """Test RNA embedding layer"""
        embedding = RNAEmbedding(config)
        input_ids = torch.randint(0, config.vocab_size, (4, 64))

        output = embedding(input_ids)

        assert output.shape == (4, 64, config.d_model)
        assert not torch.isnan(output).any()

    def test_transformer_block(self, config):
        """Test single transformer block"""
        block = TransformerBlock(config)
        hidden_states = torch.randn(4, 64, config.d_model)

        output = block(hidden_states)

        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()

    def test_transformer_block_with_mask(self, config):
        """Test transformer block with attention mask"""
        block = TransformerBlock(config)
        hidden_states = torch.randn(4, 64, config.d_model)
        key_padding_mask = torch.zeros(4, 64, dtype=torch.bool)
        key_padding_mask[:, 50:] = True  # Mask last 14 positions

        output = block(hidden_states, key_padding_mask=key_padding_mask)

        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()

    def test_encoder(self, config, batch):
        """Test full encoder"""
        encoder = GenesisRNAEncoder(config)

        output = encoder(
            batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        batch_size, seq_len = batch['input_ids'].shape
        assert output.shape == (batch_size, seq_len, config.d_model)
        assert not torch.isnan(output).any()

    def test_full_model(self, config, batch):
        """Test complete Genesis RNA model"""
        model = GenesisRNAModel(config)

        outputs = model(
            batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        batch_size, seq_len = batch['input_ids'].shape

        # Check all outputs exist
        assert 'mlm_logits' in outputs
        assert 'struct_logits' in outputs
        assert 'pair_logits' in outputs

        # Check shapes
        assert outputs['mlm_logits'].shape == (batch_size, seq_len, config.vocab_size)
        assert outputs['struct_logits'].shape == (batch_size, seq_len, config.structure_num_labels)
        assert outputs['pair_logits'].shape == (batch_size, seq_len, seq_len)

        # Check no NaNs
        assert not torch.isnan(outputs['mlm_logits']).any()
        assert not torch.isnan(outputs['struct_logits']).any()
        assert not torch.isnan(outputs['pair_logits']).any()

    def test_model_parameter_count(self, config):
        """Test model has expected number of parameters"""
        model = GenesisRNAModel(config)
        num_params = sum(p.numel() for p in model.parameters())

        # Small model should have < 20M parameters
        assert num_params < 20_000_000
        assert num_params > 100_000  # But not too small

    def test_model_save_load(self, config, tmp_path):
        """Test model save and load"""
        model = GenesisRNAModel(config)
        save_path = tmp_path / "model.pt"

        # Save
        model.save_pretrained(str(save_path))
        assert save_path.exists()

        # Load
        loaded_model = GenesisRNAModel.from_pretrained(str(save_path))

        # Check parameters are identical
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_model_gradient_flow(self, config, batch):
        """Test gradients flow through model"""
        model = GenesisRNAModel(config)

        outputs = model(
            batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        # Create dummy loss
        loss = outputs['mlm_logits'].sum()
        loss.backward()

        # Check gradients exist and are not zero
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_model_eval_mode(self, config, batch):
        """Test model in evaluation mode"""
        model = GenesisRNAModel(config)
        model.eval()

        with torch.no_grad():
            outputs1 = model(batch['input_ids'])
            outputs2 = model(batch['input_ids'])

        # In eval mode with same input, outputs should be identical
        assert torch.allclose(outputs1['mlm_logits'], outputs2['mlm_logits'])

    def test_model_different_sequence_lengths(self, config):
        """Test model handles different sequence lengths"""
        model = GenesisRNAModel(config)

        for seq_len in [32, 64, 128, 256]:
            input_ids = torch.randint(0, config.vocab_size, (2, seq_len))
            outputs = model(input_ids)

            assert outputs['mlm_logits'].shape == (2, seq_len, config.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
