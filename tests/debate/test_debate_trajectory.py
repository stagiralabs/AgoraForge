import json
import re
from pathlib import Path

import torch

from agoraforge.visualization.debate.trajectory import capture, render_html, replay_to_html
from agoraforge.envs.debate.env import BatchedDebateEnv
from agoraforge.models.factory import build_actor, build_model_config, build_shared
from agoraforge.conf.runs.debate import learned_fresh
from agoraforge.conf.schema import build_env_config


def test_protocol_debate_trajectory_is_complete_and_renderable():
    torch.manual_seed(0)
    run = learned_fresh.get_config()
    cfg = build_env_config(run, level=run.levels[0])
    cfg.max_timestep = 4
    batch = BatchedDebateEnv.from_config(cfg, batch_size=1, device=torch.device("cpu"), seeds=[3])
    actor = build_actor(build_model_config(run.actor_model, cfg, run.decoding)).eval()

    payload = capture(batch, actor, seed=9)

    assert payload["schema"] == "agoraforge.debate-trajectory.v1"
    assert len(payload["steps"]) == 4
    assert payload["steps"][-1]["state"]["judge_selected"]
    assert len(payload["steps"][-1]["state"]["protocol_state"]) == cfg.num_claims
    html = render_html(payload)
    embedded = re.search(r'id="trajectory-data" type="application/json">(.*?)</script>', html, re.S)
    assert embedded
    assert json.loads(embedded.group(1))["target"] == payload["target"]
    assert html.endswith("</html>\n")


def test_debate_replay_loads_shared_training_checkpoint(tmp_path):
    run = learned_fresh.get_config()
    cfg = build_env_config(run, level=run.levels[0])
    cfg.max_timestep = 2
    model_cfg = build_model_config(run.actor_model, cfg, run.decoding)
    model = build_shared(model_cfg).eval()
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "epoch": 3,
            "model_config": model_cfg.to_dict(),
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    out, payload = replay_to_html(
        config=str(Path(learned_fresh.__file__)),
        actor_checkpoint=str(checkpoint),
        out=str(tmp_path / "debate.html"),
        seed=2,
    )
    assert out.exists()
    assert payload["control_mode"] == "learned"
