import torch

from gb_demand_forecasting.models.tft import ResearchTemporalFusionTransformer, quantile_loss


def test_tft_forward_shape_and_loss():
    model = ResearchTemporalFusionTransformer(
        n_encoder_vars=6,
        n_decoder_vars=5,
        hidden_size=16,
        heads=2,
        dropout=0.0,
        quantiles=[0.05, 0.5, 0.95],
    )
    encoder = torch.randn(3, 96, 6)
    decoder = torch.randn(3, 72, 5)
    out = model(encoder, decoder)
    assert out.shape == (3, 72, 3)
    loss = quantile_loss(out, torch.randn(3, 72), [0.05, 0.5, 0.95])
    assert torch.isfinite(loss)
