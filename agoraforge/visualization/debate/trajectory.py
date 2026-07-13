"""Capture and render one protocol-bound debate as standalone HTML.

The viewer shows the claim graph, transcript authorship, learned protocol state,
terminal judge selection/marginals, actions, rewards, and the protocol prediction
against the unary prior and full-graph target.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import torch

from agoraforge.conf.schema import build_env_config, load_run_config, select_level
from agoraforge.envs.debate.actions import REVEAL_TYPE_TO_IDX
from agoraforge.envs.debate.env import BatchedDebateEnv
from agoraforge.envs.debate.obs import ACTOR_OBS_LEGEND
from agoraforge.envs.debate.policy import sample_actions
from agoraforge.models.graph_transformer import GraphTransformerConfig
from agoraforge.models.factory import build_shared

_TEMPLATE_PATH = Path(__file__).with_name("template.html")


def _host(t: torch.Tensor):
    return t[0].detach().cpu()


def _state(batch: BatchedDebateEnv) -> dict:
    target = int(batch.target_idx[0])
    return {
        "timestep": int(batch.timestep[0]),
        "transcript": [int(x) for x in _host(batch.slots)[_host(batch.slot_valid)].tolist()],
        "revealed_by": _host(batch.revealed_by).int().tolist(),
        "protocol_state": _host(batch.s).tolist(),
        "judge_selected": [int(x) for x in _host(batch.judge_slots)[_host(batch.judge_slot_valid)].tolist()],
        "judge_marginals": _host(batch.judge_p).tolist(),
        "prediction": float(batch.final_p[0]),
        "prior": float(batch.prior_p[0]),
        "full_target": float(batch.p_full[0, target]),
        "fitness": float(batch.fitness()[0]),
    }


def capture(batch: BatchedDebateEnv, actor, *, seed: int = 0) -> dict:
    """Run one resident-env episode and return JSON-safe viewer data."""
    if batch.B != 1:
        raise ValueError("debate trajectory capture requires batch_size=1")
    gen = torch.Generator(device=batch.device).manual_seed(seed)
    initial = _state(batch)
    steps = []
    returns = torch.zeros(batch.A)
    with torch.no_grad():
        for _ in range(batch.cfg.max_timestep):
            inputs = batch.policy_inputs()
            actions, _ = sample_actions(actor(inputs.actor_obs), inputs.masks, batch, gen)
            action_rows = []
            for a in range(batch.A):
                claim = int(actions.reveal_claim[0, a])
                signal = actions.signals[0, a]
                action_rows.append({
                    "role": "PRO" if a == 0 else "CON",
                    "type": "reveal" if int(actions.reveal_type[0, a]) == REVEAL_TYPE_TO_IDX["reveal"] else "pass",
                    "claim": claim if claim >= 0 else None,
                    "signal_l2": float(signal.norm()),
                    "active_signal_claims": int((signal.abs().sum(-1) > 0).sum()),
                })
            reward = batch.step(actions)[0, :, 0].detach().cpu()
            returns += reward
            steps.append({
                "actions": action_rows,
                "reward": reward.tolist(),
                "returns": returns.tolist(),
                "state": _state(batch),
            })

    couplings = _host(batch.couplings)
    edges = [
        {"source": i, "target": j, "coupling": float(couplings[i, j])}
        for i in range(batch.N) for j in range(i + 1, batch.N)
        if float(couplings[i, j]) != 0.0
    ]
    return {
        "schema": "agoraforge.debate-trajectory.v1",
        "title": f"Protocol-bound debate · seed {seed}",
        "roles": ["PRO", "CON"],
        "target": int(batch.target_idx[0]),
        "claims": [
            {"id": i, "unary": float(batch.unary[0, i]), "truth": int(batch.truth[0, i]),
             "p_full": float(batch.p_full[0, i])}
            for i in range(batch.N)
        ],
        "edges": edges,
        "initial": initial,
        "steps": steps,
        "actor_input_legend": ACTOR_OBS_LEGEND,
        "judge_cap": batch.J,
        "control_mode": batch.cfg.control_mode,
    }


def render_html(payload: dict) -> str:
    """Render a captured debate payload as a standalone HTML document."""
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return _TEMPLATE_PATH.read_text(encoding="utf-8").replace(
        "__TITLE__", html.escape(payload["title"])
    ).replace("__DATA__", data)


def write_html(path: str | Path, payload: dict) -> Path:
    """Write a captured debate payload as standalone HTML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(payload), encoding="utf-8")
    return path


def replay_to_html(*, config: str, actor_checkpoint: str, out: str, mechanism_params: str = "",
                   level: str = "", seed: int = 0, device: str = "cpu") -> tuple[Path, dict]:
    run = load_run_config(config)
    if run.env_name != "debate":
        raise ValueError(f"debate visualization requires a debate config, got {run.env_name!r}")
    cfg = build_env_config(run, level=select_level(run, level))
    if cfg.control_mode != "learned":
        raise ValueError("debate trajectory viewer requires a protocol-bound (learned) config")
    if mechanism_params:
        cfg.learned_mechanism_params = mechanism_params
    dev = torch.device(device)
    bundle = torch.load(actor_checkpoint, map_location=dev, weights_only=False)
    actor_cfg = GraphTransformerConfig.from_dict(bundle["model_config"])
    actor = build_shared(actor_cfg).to(dev).eval()
    actor.load_state_dict(bundle["state_dict"])
    batch = BatchedDebateEnv.from_config(cfg, batch_size=1, device=dev, seeds=[seed])
    payload = capture(batch, actor, seed=seed)
    return write_html(out, payload), payload


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Render a protocol-bound debate trajectory")
    p.add_argument("--config", required=True)
    p.add_argument("--level", default="", help="config level name (default: first)")
    p.add_argument("--actor-checkpoint", required=True)
    p.add_argument("--mechanism-params", default="", help="searched mechanism parameter file")
    p.add_argument("--out", default="/tmp/agf_debate_trajectory.html")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    out, payload = replay_to_html(**vars(args))
    final = payload["steps"][-1]["state"]
    print(f"wrote {out} | prediction={final['prediction']:.3f} full={final['full_target']:.3f} fitness={final['fitness']:.3f}")


if __name__ == "__main__":
    main()
