from types import SimpleNamespace

from agoraforge.training.metrics import EpochMetricSummary, write_epoch_scalars


class RecordingWriter:
    def __init__(self):
        self.tags = []

    def add_scalar(self, tag, value, epoch):
        self.tags.append(tag)


def test_disabled_ppo_diagnostics_do_not_log_nan_series():
    writer = RecordingWriter()
    summary = EpochMetricSummary(
        overall_return=0.0,
        overall={},
        n_transitions_global=4,
        level_return_mean={},
        level_primary_mean={},
    )
    loss_terms = {
        "actor_advantage_loss": 0.0,
        "actor_kl_loss": 0.0,
        "kl_to_reference": 0.0,
        "kl_coeff": 0.0,
        "actor_lr": 0.1,
        "critic_lr": 0.1,
        "approx_kl": float("nan"),
        "clip_fraction": float("nan"),
        "ratio_mean": float("nan"),
        "ratio_max": float("nan"),
        "critic_explained_variance": float("nan"),
    }
    write_epoch_scalars(
        writer,
        epoch=0,
        weights={},
        levels={},
        summary=summary,
        spec=SimpleNamespace(writer_scalars={}, primary_metric="metric"),
        actor_loss=0.0,
        critic_loss=0.0,
        loss_terms=loss_terms,
        advantage_mean=0.0,
        advantage_std=1.0,
    )
    assert not any(tag.endswith((
        "approx_kl", "clip_fraction", "ratio_mean", "ratio_max",
        "critic_explained_variance",
    )) for tag in writer.tags)
