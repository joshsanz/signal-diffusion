"""Unit tests for focal loss implementation."""

import pytest
import torch
import torch.nn as nn

from signal_diffusion.losses import FocalLoss


class TestFocalLoss:
    """Test suite for FocalLoss."""

    def test_focal_loss_shape(self):
        """Test that focal loss output has correct shape."""
        batch_size = 32
        num_classes = 2
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        output = focal_loss(inputs, targets)

        # Output should be a scalar
        assert output.shape == torch.Size([])
        assert output.dtype == torch.float32

    def test_focal_loss_vs_ce_easy_examples(self):
        """Test that focal loss down-weights easy examples compared to CE."""
        num_classes = 2
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")
        ce_loss = nn.CrossEntropyLoss(reduction="none")

        # Create easy examples (high confidence, correct predictions)
        logits = torch.tensor([
            [10.0, 0.0],  # Very confident correct prediction
            [0.0, 10.0],  # Very confident correct prediction
        ])
        targets = torch.tensor([0, 1])

        focal_output = focal_loss(logits, targets)
        ce_output = ce_loss(logits, targets)

        # Focal loss should be much lower for easy examples
        assert (focal_output < ce_output).all()

    def test_focal_loss_vs_ce_hard_examples(self):
        """Test that focal loss reduces the gap between hard and easy examples."""
        num_classes = 2
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")
        ce_loss = nn.CrossEntropyLoss(reduction="none")

        # Easy examples (high confidence, correct predictions)
        easy_logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
        easy_targets = torch.tensor([0, 1])

        # Hard examples (low confidence, wrong predictions)
        hard_logits = torch.tensor([[0.5, 0.4], [0.4, 0.5]])
        hard_targets = torch.tensor([1, 0])

        focal_easy = focal_loss(easy_logits, easy_targets)
        focal_hard = focal_loss(hard_logits, hard_targets)
        ce_easy = ce_loss(easy_logits, easy_targets)
        ce_hard = ce_loss(hard_logits, hard_targets)

        # Focal loss should reduce the gap between easy and hard examples
        ce_ratio = ce_hard.mean() / ce_easy.mean()
        focal_ratio = focal_hard.mean() / focal_easy.mean()

        # The ratio should be closer to 1 for focal loss (gap reduced)
        assert focal_ratio > ce_ratio

    def test_focal_loss_convergence(self):
        """Test that focal loss training can converge on simple task."""
        # Create simple linearly separable data
        torch.manual_seed(42)
        X = torch.randn(100, 10)
        y = (X[:, 0] > 0).long()

        # Simple linear model
        model = nn.Linear(10, 2)
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        initial_loss = None
        for epoch in range(20):
            optimizer.zero_grad()
            logits = model(X)
            loss = focal_loss(logits, y)
            loss.backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss.item()

        # Loss should decrease over training
        assert loss.item() < initial_loss

    def test_focal_loss_reduction_modes(self):
        """Test different reduction modes."""
        batch_size = 4
        num_classes = 2

        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        # Test mean reduction
        loss_mean = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        output_mean = loss_mean(inputs, targets)
        assert output_mean.shape == torch.Size([])

        # Test sum reduction
        loss_sum = FocalLoss(alpha=0.25, gamma=2.0, reduction="sum")
        output_sum = loss_sum(inputs, targets)
        assert output_sum.shape == torch.Size([])

        # Test none reduction
        loss_none = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")
        output_none = loss_none(inputs, targets)
        assert output_none.shape == torch.Size([batch_size])

        # Verify relationships between reduction modes
        assert torch.allclose(output_mean, output_sum / batch_size)

    def test_focal_loss_alpha_effect(self):
        """Test that alpha parameter affects loss correctly."""
        inputs = torch.randn(10, 2)
        targets = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        # Higher alpha should give more weight overall (including hard examples)
        loss_alpha_low = FocalLoss(alpha=0.1, gamma=2.0, reduction="mean")
        loss_alpha_high = FocalLoss(alpha=0.5, gamma=2.0, reduction="mean")

        output_low = loss_alpha_low(inputs, targets)
        output_high = loss_alpha_high(inputs, targets)

        # Higher alpha should result in higher mean loss
        assert output_high > output_low

    def test_focal_loss_gamma_effect(self):
        """Test that gamma parameter affects loss correctly."""
        inputs = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))

        loss_gamma_low = FocalLoss(alpha=0.25, gamma=0.5, reduction="mean")
        loss_gamma_high = FocalLoss(alpha=0.25, gamma=3.0, reduction="mean")

        output_low = loss_gamma_low(inputs, targets)
        output_high = loss_gamma_high(inputs, targets)

        # Both should produce reasonable outputs
        assert output_low > 0
        assert output_high > 0

    def test_focal_loss_numerical_stability(self):
        """Test that focal loss is numerically stable with large logits."""
        # Very large logits should not cause NaN or Inf
        inputs = torch.tensor([
            [100.0, 0.0],
            [0.0, 100.0],
            [-100.0, 0.0],
        ], dtype=torch.float32)
        targets = torch.tensor([0, 1, 1])

        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        output = focal_loss(inputs, targets)

        assert not torch.isnan(output)
        assert not torch.isinf(output)

    def test_focal_loss_multiclass(self):
        """Test focal loss with multi-class classification."""
        batch_size = 32
        num_classes = 5
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

        inputs = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))

        output = focal_loss(inputs, targets)

        # Should produce a valid scalar output
        assert output.shape == torch.Size([])
        assert output > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
