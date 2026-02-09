"""Tests for model architectures."""

import torch

from articulated.estimation.model import GRU, LSTM, RNN, StateEstimationModel


class TestRNN:
    """Tests for the RNN architecture."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shapes."""
        batch_size = 8
        seq_length = 50
        input_size = 6
        hidden_size = 128
        output_size = 64

        model = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
        )

        x = torch.randn(batch_size, seq_length, input_size)
        output, hidden_states = model(x)

        assert output.shape == (batch_size, seq_length, output_size)
        assert hidden_states.shape == (batch_size, seq_length, hidden_size)

    def test_dropout(self):
        """Test that dropout is applied during training."""
        model = RNN(input_size=6, hidden_size=64, output_size=32, dropout=0.5)
        model.train()
        x = torch.randn(4, 20, 6)
        out1, _ = model(x)
        out2, _ = model(x)
        # With dropout=0.5, outputs should differ in train mode
        assert not torch.allclose(out1, out2)


class TestLSTM:
    """Tests for LSTM architecture."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shapes."""
        batch_size = 8
        seq_length = 50
        input_size = 6
        hidden_size = 128
        output_size = 64

        model = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
        )

        x = torch.randn(batch_size, seq_length, input_size)
        output, hidden_states = model(x)

        assert output.shape == (batch_size, seq_length, output_size)
        assert hidden_states.shape == (batch_size, seq_length, hidden_size)

    def test_multi_layer(self):
        """Test LSTM with multiple layers."""
        model = LSTM(input_size=6, hidden_size=64, output_size=32, num_layers=2)
        x = torch.randn(4, 20, 6)
        output, hidden_states = model(x)

        assert output.shape == (4, 20, 32)
        assert hidden_states.shape == (4, 20, 64)


class TestGRU:
    """Tests for GRU architecture."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shapes."""
        batch_size = 8
        seq_length = 50
        input_size = 6
        hidden_size = 128
        output_size = 64

        model = GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
        )

        x = torch.randn(batch_size, seq_length, input_size)
        output, hidden_states = model(x)

        assert output.shape == (batch_size, seq_length, output_size)
        assert hidden_states.shape == (batch_size, seq_length, hidden_size)

    def test_multi_layer(self):
        """Test GRU with multiple layers."""
        model = GRU(input_size=6, hidden_size=64, output_size=32, num_layers=2)
        x = torch.randn(4, 20, 6)
        output, hidden_states = model(x)

        assert output.shape == (4, 20, 32)
        assert hidden_states.shape == (4, 20, 64)


class TestStateEstimationModel:
    """Tests for the Lightning wrapper."""

    def test_forward_with_rnn(self):
        """Test forward pass with RNN backend."""
        model = StateEstimationModel(
            input_size=6,
            hidden_size=128,
            output_size=64,
            model_type="rnn",
            use_init_pos=False,
        )

        x = torch.randn(4, 20, 6)
        output, hidden_states = model(x)

        assert output.shape == (4, 20, 64)
        assert hidden_states.shape == (4, 20, 128)

    def test_forward_with_dropout(self):
        """Test forward pass with dropout enabled."""
        model = StateEstimationModel(
            input_size=6,
            hidden_size=128,
            output_size=64,
            model_type="gru",
            use_init_pos=False,
            dropout=0.5,
        )

        x = torch.randn(4, 20, 6)
        output, hidden_states = model(x)

        assert output.shape == (4, 20, 64)
        assert hidden_states.shape == (4, 20, 128)

    def test_forward_with_init_pos(self):
        """Test forward pass with initial position encoding."""
        model = StateEstimationModel(
            input_size=6,
            hidden_size=128,
            output_size=64,
            model_type="rnn",
            use_init_pos=True,
        )

        x = torch.randn(4, 20, 6)
        init_pc = torch.randn(4, 64)
        h0 = model._encode_init_pos(init_pc)
        output, hidden_states = model(x, hidden=h0)

        assert output.shape == (4, 20, 64)
        assert hidden_states.shape == (4, 20, 128)

    def test_forward_lstm_with_init_pos(self):
        """Test LSTM forward pass with initial position encoding."""
        model = StateEstimationModel(
            input_size=6,
            hidden_size=128,
            output_size=64,
            model_type="lstm",
            use_init_pos=True,
        )

        x = torch.randn(4, 20, 6)
        init_pc = torch.randn(4, 64)
        h0 = model._encode_init_pos(init_pc)
        output, hidden_states = model(x, hidden=h0)

        assert output.shape == (4, 20, 64)
        assert hidden_states.shape == (4, 20, 128)

    def test_get_embedding(self):
        """Test embedding extraction for RL."""
        model = StateEstimationModel(
            input_size=6,
            hidden_size=128,
            output_size=64,
            model_type="rnn",
            use_init_pos=False,
        )

        x = torch.randn(4, 20, 6)
        embedding = model.get_embedding(x)

        assert embedding.shape == (4, 128)

    def test_get_hidden_states(self):
        """Test hidden state extraction for analysis."""
        model = StateEstimationModel(
            input_size=6,
            hidden_size=128,
            output_size=64,
            model_type="rnn",
            use_init_pos=False,
        )

        x = torch.randn(4, 20, 6)
        hidden_states = model.get_hidden_states(x)

        assert hidden_states.shape == (4, 20, 128)
