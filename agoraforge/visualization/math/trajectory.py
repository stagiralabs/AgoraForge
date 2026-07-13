"""Capture and render a training-faithful math trajectory.

The four mechanism-backed arms (market / bounty_only / collaborative / learned)
train on the resident tensor env ``agoraforge.envs.math.env.BatchedMathEnv`` with
a per-agent ``obs_formula_cap`` LRU theorem-context (K=16). To keep the viewer's
per-agent actor-input graph faithful to that cap, those arms must be replayed
through the resident env itself: this module runs one
B=1 resident episode, captures the *exact* capped observation fed to the actor
each step, and renders it as a standalone interactive page.

``centralized`` also trains on the resident env but is always uncapped -- full
visibility is the planner baseline's defining property, so
``select_topk_formulas`` returns ``None`` in that mode and the replay reproduces
the full library the planner conditioned on. Its return is the accumulated
fitness reward rather than an economic value.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np
import torch

from agoraforge.conf.schema import build_env_config, load_run_config, select_level
from agoraforge.models.graph_transformer import GraphTransformerConfig
from agoraforge.models.factory import build_shared
from agoraforge.envs.math.obs import (
    ACTOR_OBS_LEGEND,
    AGENT_LAST_TURN,
    AGENT_TIMESTEP,
    LEARNED_PROVING_COLS,
)
from agoraforge.envs.math.policy import sample_actions, to_env_actions
from agoraforge.envs.math.env import BatchedMathEnv
from agoraforge.envs.math.actions import CONJ_MODES, MATH_ACTION_TYPES

# Mechanism-backed arms: their observation appends the agent's learned per-formula
# state (s). ``centralized`` also replays on the resident env but uses the standard
# formula-feature layout, so it is handled separately.
_TEMPLATE_PATH = Path(__file__).with_name("template.html")


# Human-readable names for the 15 math/knowledge feature columns kept in the
# learned/market observation, in LEARNED_PROVING_COLS order.
_PROVING_COL_NAMES = {
    0: "local_proven",
    1: "public_concrete",
    2: "public_proven",
    3: "job_active_target",
    4: "job_tau_remaining",
    5: "job_type_prove",
    6: "job_type_conj",
    7: "cumulative_proof",
    8: "cumulative_conj",
    9: "query_prob_time_known",
    10: "query_probability",
    11: "query_time_ratio",
    12: "query_related_in_known",
    13: "query_related_out_known",
    14: "formula_is_target",
}


def _actor_input_legend(batch: BatchedMathEnv) -> dict:
    """Feature-name legend matching the resident observation layout.

    ``formula_features`` is ``[15 proving cols] + s(d_state)``; the three
    ``agent_scalars`` are economic-value, timestep ratio and last-turn.
    """
    d_state = int(batch.dims.d_state)
    formula_features = [_PROVING_COL_NAMES[c] for c in LEARNED_PROVING_COLS]
    formula_features += [f"s[{i}]" for i in range(d_state)]
    agent_scalars = [None] * 3
    agent_scalars[0] = "economic_value/initial_cash"
    agent_scalars[AGENT_TIMESTEP] = "timestep/max_timestep"
    agent_scalars[AGENT_LAST_TURN] = "is_last_turn"
    return {"formula_features": formula_features, "agent_scalars": agent_scalars}


def _formula_label(phi: int, num_theorems: int) -> str:
    theorem_id = int(phi) % num_theorems
    sign = 0 if int(phi) < num_theorems else 1
    return f"¬φ{theorem_id}" if sign else f"φ{theorem_id}"


def _formula_nodes(cfg, num_theorems: int) -> list[dict]:
    nodes = []
    cols = max(1, math.ceil(math.sqrt(num_theorems)))
    x_spacing = 260
    y_spacing = 190
    pair_gap = 72
    for theorem_id in range(num_theorems):
        col = theorem_id % cols
        row = theorem_id // cols
        center_x = 120 + col * x_spacing
        center_y = 110 + row * y_spacing
        for sign in (0, 1):
            phi = theorem_id + sign * num_theorems
            nodes.append(
                {
                    "phi": phi,
                    "theorem_id": theorem_id,
                    "label": _formula_label(phi, num_theorems),
                    "truth": bool(cfg.truth_map[theorem_id] == sign),
                    "x": float(center_x + (-pair_gap / 2 if sign == 0 else pair_gap / 2)),
                    "y": float(center_y),
                }
            )
    return nodes


def _utility_edges(cfg) -> list[dict]:
    return [
        {
            "source": int(src_phi),
            "target": int(dst_phi),
            "weight": float(weight),
        }
        for (src_phi, dst_phi), weight in sorted(cfg.utility_weights.items())
    ]


def _target_theorems(trajectory: dict, num_theorems: int) -> list[int]:
    """Theorems designated as targets in the captured rollout."""
    public = trajectory["initial_state"]["public_library"]
    concrete = {int(phi) for phi in public.get("concrete", [])}
    resolved = {int(phi) for phi in public.get("resolved", [])}
    targets = set()
    for phi in concrete:
        if phi in resolved:
            continue
        if (phi + num_theorems) % (2 * num_theorems) in resolved:
            continue
        targets.add(phi % num_theorems)
    return sorted(targets)


def _viewer_job(job: dict | None, num_theorems: int) -> dict | None:
    if job is None:
        return None
    target = int(job["target"])
    return {
        "type": job["type"],
        "target_raw": target,
        "target": _formula_label(target, num_theorems),
        "mode": job.get("mode"),
        "tau_rem": int(job["tau_rem"]),
        "tau_eff": int(job["tau_eff"]),
    }


def _agent_colors(num_agents: int) -> list[str]:
    palette = [
        "#0f766e",
        "#c2410c",
        "#2563eb",
        "#8b5cf6",
        "#65a30d",
        "#db2777",
        "#0891b2",
        "#b45309",
    ]
    return [palette[idx % len(palette)] for idx in range(num_agents)]


def _viewer_payload(trajectory: dict, *, cfg, label: str | None = None) -> dict:
    steps = trajectory.get("steps", [])
    initial_state = trajectory["initial_state"]
    num_agents = len(initial_state.get("agents", []))
    num_theorems = max(int(cfg.F_size) // 2, 1)

    return_series = [[0.0] for _ in range(num_agents)]
    for step in steps:
        cumulative = step.get("cumulative_returns", [0.0] * num_agents)
        for agent_id in range(num_agents):
            return_series[agent_id].append(float(cumulative[agent_id]))
    return_mean = [
        float(sum(series[t] for series in return_series) / max(num_agents, 1))
        for t in range(len(return_series[0]) if return_series else 0)
    ]

    frames = []
    current_snapshot = initial_state
    for frame_index in range(len(steps) + 1):
        current_step = steps[frame_index] if frame_index < len(steps) else None
        if frame_index > 0:
            current_snapshot = steps[frame_index - 1]["state_after"]

        private_concrete_agents = []
        private_resolved_agents = []
        for agent_id in range(num_agents):
            library = current_snapshot["agents"][agent_id]["library"]
            private_concrete_agents.append(
                sorted(int(phi) for phi in library.get("concrete", []))
            )
            private_resolved_agents.append(
                sorted(int(phi) for phi in library.get("resolved", []))
            )

        frames.append(
            {
                "public_concrete": sorted(
                    int(phi)
                    for phi in current_snapshot["public_library"].get("concrete", [])
                ),
                "public_resolved": sorted(
                    int(phi)
                    for phi in current_snapshot["public_library"].get("resolved", [])
                ),
                "private_concrete_agents": private_concrete_agents,
                "private_resolved_agents": private_resolved_agents,
                "jobs": [
                    _viewer_job(job, num_theorems)
                    for job in current_snapshot.get("jobs", [None] * num_agents)
                ],
                "action_types": [
                    action.get("type") if isinstance(action, dict) else None
                    for action in (
                        current_step.get("actions", [])
                        if current_step is not None
                        else []
                    )
                ],
                "actor_inputs": (
                    current_step.get("actor_inputs", [])
                    if current_step is not None
                    else []
                ),
            }
        )

    return {
        "title": label or "Math trajectory",
        "num_agents": num_agents,
        "num_theorems": num_theorems,
        "formula_nodes": _formula_nodes(cfg, num_theorems),
        "utility_edges": _utility_edges(cfg),
        "agent_colors": _agent_colors(num_agents),
        "actor_input_legend": trajectory.get("actor_input_legend", {}),
        "frames": frames,
        "series": {"returns": return_series, "return_mean": return_mean},
        "centralized": bool(trajectory.get("centralized", False)),
        "control_mode": trajectory["control_mode"],
        "target_theorems": _target_theorems(trajectory, num_theorems),
    }


class _CaptureBuffer:
    """Capture the per-step state the interactive viewer needs (cheap at B=1)."""

    def __init__(self, batch: BatchedMathEnv):
        self.b = batch
        self.states = []        # post-step state snapshots (initial + one per step)
        self.actions = []       # per-step actions (one per transition)
        self.actor_inputs = []  # per-step actor obs (one per transition)
        self.rewards = []       # per-step per-agent reward (one per transition)

    def _np(self, t):
        return t[0].detach().cpu().numpy()

    def snapshot_state(self):
        b = self.b
        self.states.append(
            {
                "public_concrete": self._np(b.public_concrete).astype(bool).copy(),
                "public_resolved": self._np(b.public_resolved).astype(bool).copy(),
                "agent_concrete": self._np(b.concrete).astype(bool).copy(),   # [A, F]
                "agent_resolved": self._np(b.resolved).astype(bool).copy(),   # [A, F]
                "econ_value": self._np(b.economic_value()).astype(float).copy(),  # [A]
                "job_type": self._np(b.job_type).astype(int).copy(),          # [A]
                "job_target": self._np(b.job_target).astype(int).copy(),      # [A]
                "job_tau": self._np(b.job_tau).astype(int).copy(),            # [A]
                "job_mode": self._np(b.job_mode).astype(int).copy(),          # [A]
            }
        )

    def snapshot_actions(self, env_actions):
        self.actions.append(
            {
                "math_type": env_actions.math_type[0].detach().cpu().numpy().copy(),
                "math_formula": env_actions.math_formula[0].detach().cpu().numpy().copy(),
            }
        )

    def snapshot_actor_obs(self, actor_obs):
        # actor_obs is BA-flattened (B=1) so row a is agent a. None means the
        # caller dropped the per-agent actor-network payload (heavy at large N).
        if actor_obs is None:
            self.actor_inputs.append(None)
            return
        keys = ("formula_features", "formula_mask", "formula_ids",
                "neg_formula_ids", "agent_scalars", "query_related_edges")
        host = {k: actor_obs[k].detach().cpu().numpy() for k in keys if k in actor_obs}
        self.actor_inputs.append(host)

    def snapshot_reward(self, rewards):
        self.rewards.append(rewards[0, :, 0].detach().cpu().numpy().copy())


def _capture_buffers(
    batch,
    actor,
    *,
    generator: torch.Generator,
    capture_actor_inputs: bool = True,
) -> _CaptureBuffer:
    """One fresh B=1 resident episode, capturing the obs the actor saw.

    ``capture_actor_inputs`` toggles the per-agent actor-network payload, which
    the uncapped centralized planner can blow up at large N; the global view and
    returns are unaffected when it is dropped.
    """
    cap = _CaptureBuffer(batch)
    cap.snapshot_state()
    T = int(batch.cfg.max_timestep)
    with torch.no_grad():
        for _ in range(T):
            sel = batch.select_topk_formulas()
            actor_obs = batch.actor_obs(sel)
            masks = batch.available_action_masks(sel)
            logits = actor(actor_obs)
            actions, _ = sample_actions(logits, masks, batch, generator)
            env_actions = actions if sel is None else to_env_actions(actions, sel, batch)
            cap.snapshot_actor_obs(actor_obs if capture_actor_inputs else None)
            cap.snapshot_actions(env_actions)
            rewards = batch.step(env_actions)
            cap.snapshot_reward(rewards)
            cap.snapshot_state()
    return cap


# --- resident state -> viewer trajectory schema -----------------------------


def _phi_list(mask_row) -> list:
    return [int(phi) for phi in np.nonzero(mask_row)[0]]


def _library(concrete_row, resolved_row) -> dict:
    return {"concrete": _phi_list(concrete_row), "resolved": _phi_list(resolved_row)}


def _job_dict(state, a):
    jt = int(state["job_type"][a])
    if jt == BatchedMathEnv.JOB_NONE:
        return None
    target = int(state["job_target"][a])
    if target < 0:
        return None
    job_type = "prove" if jt == BatchedMathEnv.JOB_PROVE else "conj"
    tau_rem = int(state["job_tau"][a])
    return {
        "type": job_type,
        "target": target,
        "mode": CONJ_MODES[int(state["job_mode"][a])] if job_type == "conj" else None,
        "tau_rem": tau_rem,
        "tau_eff": max(tau_rem, 1),
    }


def _state_snapshot(state, A) -> dict:
    agents = [
        {
            "library": _library(state["agent_concrete"][a], state["agent_resolved"][a]),
        }
        for a in range(A)
    ]
    return {
        "public_library": _library(state["public_concrete"], state["public_resolved"]),
        "agents": agents,
        "jobs": [_job_dict(state, a) for a in range(A)],
    }


def _actor_inputs_frame(obs) -> list:
    """One actor-input record per agent for the actor-network panel."""
    if obs is None:
        return []
    A = obs["formula_ids"].shape[0]
    frame = []
    for a in range(A):
        frame.append(
            {
                "formula_ids": [int(x) for x in obs["formula_ids"][a]],
                "neg_formula_ids": [int(x) for x in obs["neg_formula_ids"][a]],
                "formula_mask": [float(x) for x in obs["formula_mask"][a]],
                "formula_features": [[float(v) for v in row] for row in obs["formula_features"][a]],
                "agent_scalars": [float(x) for x in obs["agent_scalars"][a]],
                "query_related_edges": [[float(v) for v in row] for row in obs["query_related_edges"][a]],
            }
        )
    return frame


def _build_trajectory(cap: _CaptureBuffer, batch: BatchedMathEnv) -> dict:
    A = batch.A
    centralized = batch.centralized
    initial_state = _state_snapshot(cap.states[0], A)
    init_econ = cap.states[0]["econ_value"]
    running = [0.0] * A
    steps = []
    for i, env_actions in enumerate(cap.actions):
        state_after = cap.states[i + 1]
        if centralized:
            # The planner's return is the accumulated fitness reward, not an
            # economic value (econ value is identically zero in centralized mode).
            running = [running[a] + float(cap.rewards[i][a]) for a in range(A)]
            cumulative = list(running)
        else:
            cumulative = [float(state_after["econ_value"][a] - init_econ[a]) for a in range(A)]
        action_list = []
        for a in range(A):
            t_idx = int(env_actions["math_type"][a])
            action_list.append(
                {
                    "type": MATH_ACTION_TYPES[t_idx] if 0 <= t_idx < len(MATH_ACTION_TYPES) else None,
                    "formula": int(env_actions["math_formula"][a]) if env_actions["math_formula"][a] >= 0 else None,
                }
            )
        steps.append(
            {
                "state_after": _state_snapshot(state_after, A),
                "actions": action_list,
                "cumulative_returns": cumulative,
                "actor_inputs": _actor_inputs_frame(cap.actor_inputs[i]),
            }
        )

    return {
        "centralized": centralized,
        "control_mode": batch.cfg.control_mode,
        "initial_state": initial_state,
        "steps": steps,
        # The centralized planner uses the standard formula-feature layout; the
        # mechanism-backed arms append the learned market/agent latents.
        "actor_input_legend": (
            ACTOR_OBS_LEGEND if centralized else _actor_input_legend(batch)
        ),
    }


class _ViewerConfig:
    """Expose sampled graph data needed to lay out the viewer.

    truth_map / utility_weights live on the constructed batch (they are sampled
    per-instance), not on the static config, so we read them back from tensors.
    """

    def __init__(self, batch: BatchedMathEnv):
        N = batch.N
        self.F_size = batch.F
        truth = batch.truth[0].detach().cpu().numpy()  # [F]; truth[phi]==1 -> phi is the true member
        # truth_map[t] = sign of the true member: 0 if phi_t true, else 1.
        self.truth_map = {t: (0 if truth[t] > 0.5 else 1) for t in range(N)}
        w = batch.graph_weights[0].detach().cpu().numpy()  # [F, F]
        self.utility_weights = {
            (int(s), int(d)): float(w[s, d])
            for s in range(batch.F)
            for d in range(batch.F)
            if w[s, d] != 0.0
        }


def capture(
    batch: BatchedMathEnv,
    actor,
    *,
    seed: int = 0,
    capture_actor_inputs: bool = True,
    label: str | None = None,
) -> dict:
    """Run one episode and return the JSON-safe math viewer payload."""
    if batch.B != 1:
        raise ValueError("math trajectory capture requires batch_size=1")
    generator = torch.Generator(device=batch.device).manual_seed(seed)
    buffers = _capture_buffers(
        batch,
        actor,
        generator=generator,
        capture_actor_inputs=capture_actor_inputs,
    )
    trajectory = _build_trajectory(buffers, batch)
    return _viewer_payload(
        trajectory,
        cfg=_ViewerConfig(batch),
        label=label,
    )


def render_html(payload: dict) -> str:
    """Render a captured math payload as a standalone HTML document."""
    page_title = html.escape(payload["title"])
    payload_json = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    return (
        _TEMPLATE_PATH.read_text(encoding="utf-8")
        .replace("__PAGE_TITLE__", page_title)
        .replace("__PAGE_HEADING__", page_title)
        .replace("__PAYLOAD_JSON__", payload_json)
    )


def write_html(path: str | Path, payload: dict) -> Path:
    """Write a captured math payload as standalone HTML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(payload), encoding="utf-8")
    return path


def replay_to_html(
    *,
    config: str,
    actor_checkpoint: str,
    out: str,
    level: str = "",
    mechanism_params: str = "",
    seed: int = 0,
    capture_actor_inputs: bool = True,
    device: str = "cpu",
) -> tuple[Path, dict]:
    """Load one trained actor, replay it once, and write the visualization."""
    run = load_run_config(config)
    if run.env_name != "math":
        raise ValueError(f"math visualization requires a math config, got {run.env_name!r}")
    cfg = build_env_config(run, level=select_level(run, level))
    if mechanism_params:
        if cfg.control_mode != "learned":
            raise ValueError("--mechanism-params requires a learned math mechanism")
        cfg.learned_mechanism_params = mechanism_params
    dev = torch.device(device)
    bundle = torch.load(actor_checkpoint, map_location=dev, weights_only=False)
    model_cfg = GraphTransformerConfig.from_dict(bundle["model_config"])
    actor = build_shared(model_cfg).to(dev).eval()
    actor.load_state_dict(bundle["state_dict"])
    batch = BatchedMathEnv.from_config(cfg, batch_size=1, device=dev, seeds=[seed])
    cap = "uncapped" if batch.centralized else cfg.obs_formula_cap
    label = (
        f"Math trajectory: arm={cfg.control_mode} | N={cfg.num_theorems} "
        f"A={cfg.n_agents} | cap={cap} | epoch={bundle.get('epoch')} | seed={seed}"
    )
    payload = capture(
        batch,
        actor,
        seed=seed,
        capture_actor_inputs=capture_actor_inputs,
        label=label,
    )
    return write_html(out, payload), payload


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="training run config")
    p.add_argument("--level", default="", help="config level name (default: first)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--actor-checkpoint", required=True, help="self-describing model bundle")
    p.add_argument("--mechanism-params", default="", help="learned mechanism flat-params .pt (learned arm)")
    p.add_argument("--actor-inputs", choices=["auto", "on", "off"], default="auto",
                   help="centralized only: per-agent actor-network graph payload; 'auto' drops "
                        "it above --actor-inputs-max-theorems.")
    p.add_argument("--actor-inputs-max-theorems", type=int, default=64)
    p.add_argument("--out", default="/tmp/agf_trajectory.html")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    run = load_run_config(args.config)
    cfg = build_env_config(run, level=select_level(run, args.level))
    centralized = cfg.control_mode == "centralized"
    capture_actor_inputs = (
        cfg.num_theorems <= args.actor_inputs_max_theorems
        if centralized and args.actor_inputs == "auto"
        else args.actor_inputs != "off" or not centralized
    )
    out, payload = replay_to_html(
        config=args.config,
        actor_checkpoint=args.actor_checkpoint,
        out=args.out,
        level=args.level,
        mechanism_params=args.mechanism_params,
        seed=args.seed,
        capture_actor_inputs=capture_actor_inputs,
        device=args.device,
    )
    final_returns = [series[-1] for series in payload["series"]["returns"]]
    final_resolved = len(payload["frames"][-1]["public_resolved"])
    print(
        f"wrote {out} | returns={final_returns} | "
        f"public_resolved={final_resolved} | actor_inputs={capture_actor_inputs}"
    )


if __name__ == "__main__":
    main()
