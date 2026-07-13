"""End-to-end math trajectory capture and checkpoint replay.

Pins that each arm is replayed through the env it trained on, that the recorded
per-agent actor-input graph is faithful: resident arms (market/bounty_only/
collaborative/learned) reproduce the K=16 obs cap; centralized stays uncapped.

N=64 is the protocol floor and still has F=128 > obs_formula_cap=16, so the
cap engages; max_timestep is shortened from the scaling default purely to keep
the test fast.
"""
import json
import re
from pathlib import Path

import torch

from agoraforge.visualization.math.trajectory import (
    capture,
    render_html,
    replay_to_html,
)
from agoraforge.conf.runs.math import default as math_default
from agoraforge.conf.schema import build_env_config, run_config
from agoraforge.envs.math.scaling import scale_env_overrides
from agoraforge.models.factory import build_model_config, build_shared
from agoraforge.envs.math.env import BatchedMathEnv

N = 64
STEPS = 5


def _test_config(arm):
    run = run_config(env="math")
    for key, value in scale_env_overrides(N).items():
        run.env[key] = value
    run.env.control_mode = arm
    if arm in {"bounty_only", "collaborative"}:
        run.env.first_prover_bonus = 3.0
    if arm == "collaborative":
        run.env.prover_bonus_targets_only = True
    cfg = build_env_config(run, level=run.levels[0])
    cfg.max_timestep = STEPS  # keep the capture test fast
    return run, cfg


def _resident_payload(arm):
    torch.manual_seed(0)
    run, cfg = _test_config(arm)
    device = torch.device("cpu")
    batch = BatchedMathEnv.from_config(cfg, batch_size=1, device=device, seeds=[0])
    actor = build_shared(build_model_config(run.actor_model, cfg, run.decoding)).to(device).eval()
    return capture(batch, actor, label="test")


def test_resident_market_replay_is_capped_and_renderable():
    payload = _resident_payload("market")
    assert len(payload["frames"]) == STEPS + 1  # initial + STEPS
    assert payload["centralized"] is False
    # The actor-input graph must show no more than the K=16 trained window.
    rows = [len(ai["formula_ids"]) for fr in payload["frames"] for ai in fr["actor_inputs"]]
    assert rows and max(rows) <= 16
    html = render_html(payload)
    assert html.strip().endswith("</html>")
    embedded = re.search(r'application/json">(.*?)</script>', html, re.S)
    assert json.loads(embedded.group(1))["num_theorems"] == N


def test_centralized_replay_is_uncapped():
    payload = _resident_payload("centralized")
    assert payload["centralized"] is True
    # The planner has full visibility: its actor-input graph exceeds the K=16
    # cap that bounds the mechanism-backed arms (F=128 formulas at N=64).
    rows = [len(ai["formula_ids"]) for fr in payload["frames"] for ai in fr["actor_inputs"]]
    assert rows and max(rows) > 16
    # Team reward: every agent shares the same cumulative return each step.
    series = payload["series"]["returns"]
    for timestep in range(len(series[0])):
        assert len({agent[timestep] for agent in series}) == 1


def test_replay_to_html_round_trip(tmp_path):
    torch.manual_seed(0)
    run = math_default.get_config()
    cfg = build_env_config(run, level=run.levels[0])
    actor_cfg = build_model_config(run.actor_model, cfg, run.decoding)
    actor = build_shared(actor_cfg).eval()
    ckpt = tmp_path / "model.pt"
    torch.save(
        {
            "epoch": 7,
            "model_config": actor_cfg.to_dict(),
            "state_dict": actor.state_dict(),
        },
        ckpt,
    )

    out, payload = replay_to_html(
        config=str(Path(math_default.__file__)),
        actor_checkpoint=str(ckpt),
        out=str(tmp_path / "replay.html"),
        seed=0,
        device="cpu",
    )

    assert out.exists()
    assert "epoch=7" in payload["title"]
    assert payload["num_theorems"] == cfg.num_theorems
    assert "Math trajectory" in out.read_text()
