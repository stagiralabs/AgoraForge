"""Math environment dynamics and observation tests."""

import torch
import numpy as np

from agoraforge.conf.schema import build_env_config
from agoraforge.conf.runs.math.default import get_config
from agoraforge.envs.math.env import BatchedMathEnv, TensorActions
from agoraforge.envs.math.obs import (
    AGENT_LAST_TURN,
    AGENT_TIMESTEP,
    FORMULA_IS_TARGET,
    JOB_ACTIVE_TARGET,
    JOB_TAU_REMAINING,
    JOB_TYPE_PROVE,
    PUBLIC_CONCRETE,
    PUBLIC_PROVEN,
    QUERY_PROBABILITY,
    QUERY_PROB_TIME_KNOWN,
)
from tests.math.reference import ConjectureKernel, FormulaGraph, Library, ProofKernel, StaticQueryModel


def _batch(mode="learned", n=3, seed=3, device="cpu", **cfg_over):
    cfg = get_config()
    vcfg = build_env_config(cfg, level=cfg.levels[0])
    vcfg.control_mode = mode
    for k, v in cfg_over.items():
        setattr(vcfg, k, v)
    seeds = [seed + i for i in range(n)]
    return BatchedMathEnv.from_config(vcfg, batch_size=n, device=torch.device(device), seeds=seeds)


def _noop_actions(gpu):
    zeros_ba = torch.zeros((gpu.B, gpu.A), dtype=torch.long)
    d_action = gpu.dims.d_action if gpu.dims is not None else int(gpu.cfg.action_dim)
    return TensorActions(
        math_type=zeros_ba - 1,
        math_formula=zeros_ba - 1,
        budget=zeros_ba.to(gpu.dtype),
        math_mode=zeros_ba,
        publish_statement=torch.zeros((gpu.B, gpu.A, gpu.F), dtype=torch.bool),
        publish_proof=torch.zeros((gpu.B, gpu.A, gpu.F), dtype=torch.bool),
        market_action=torch.zeros((gpu.B, gpu.A, gpu.F, d_action), dtype=gpu.dtype),
    )


def _install_reference_world(batch, weights: dict[tuple[int, int], float]):
    batch.truth.zero_()
    batch.truth[:, : batch.N] = 1.0
    batch.query_truth = batch.truth.unsqueeze(1).expand(batch.B, batch.A, batch.F).clone()
    batch.graph_weights.zero_()
    for (src, dst), weight in weights.items():
        batch.graph_weights[:, int(src), int(dst)] = float(weight)
    src = torch.tensor([src for src, _ in weights], dtype=torch.long, device=batch.device)
    dst = torch.tensor([dst for _, dst in weights], dtype=torch.long, device=batch.device)
    batch._qw_src = src
    batch._qw_dst = dst
    batch.query_values = batch.graph_weights[:, src, dst].unsqueeze(1).expand(
        batch.B, batch.A, src.numel()
    ).clone()

    batch.concrete.zero_()
    batch.resolved.zero_()
    batch.public_concrete.zero_()
    batch.public_resolved.zero_()

    def add_concrete(phi: int):
        neg = int(batch.neg_ids[phi])
        batch.concrete[..., phi] = True
        batch.concrete[..., neg] = True
        batch.public_concrete[:, phi] = True
        batch.public_concrete[:, neg] = True

    def add_resolved(phi: int):
        add_concrete(phi)
        batch.resolved[..., phi] = True
        batch.public_resolved[:, phi] = True

    add_resolved(0)
    add_resolved(1)
    add_concrete(3)

    graph = FormulaGraph(
        batch.N,
        truth_map=np.zeros(batch.N, dtype=np.int64),
        utility_weights=weights,
    )
    library = Library(batch.F)
    library.add_resolved(0)
    library.add_resolved(1)
    library.add_concrete(3)
    return graph, library


def test_every_control_mode_steps():
    # Every mode must survive a real step(): the learned and closed-form arms
    # take different reward paths through _advance_markets.
    for mode in ("market", "bounty_only", "collaborative", "centralized", "learned"):
        gpu = _batch(mode=mode, n=2, seed=11)
        rewards = gpu.step(_noop_actions(gpu))
        assert torch.isfinite(rewards).all(), mode


def test_explicit_utility_weights_do_not_require_truth_override():
    batch = _batch(mode="market", n=2, utility_weights={(0, 1): 0.75})
    expected = torch.zeros_like(batch.graph_weights)
    expected[:, 0, 1] = 0.75
    assert torch.equal(batch.graph_weights, expected)


def test_env_math_matches_cpu_reference_kernels():
    gpu = _batch(mode="centralized", n=1, seed=30, rho=1.7, eta=2.3, horizon_H=7)
    weights = {
        (0, 3): 0.4,
        (1, 3): 0.6,
        (4, 3): 0.8,
        (5, 3): 0.2,
        (3, 6): 0.5,
    }
    graph, library = _install_reference_world(gpu, weights)
    target = torch.full((gpu.B, gpu.A), 3, dtype=torch.long)

    gw_col = gpu.graph_weights.gather(
        2, target.unsqueeze(1).expand(gpu.B, gpu.F, gpu.A)
    ).transpose(1, 2)
    support = (gpu.resolved.to(gpu.dtype) * gw_col).sum(dim=-1)
    truth = gpu._gather_baf(gpu.truth.unsqueeze(1).expand(gpu.B, gpu.A, gpu.F), target)
    env_proof_p = truth * (1.0 - (1.0 + support).pow(-float(gpu.cfg.rho)))
    ref_proof_p = ProofKernel(gpu.cfg.rho).success_probability(graph, library, 3)
    torch.testing.assert_close(env_proof_p[0, 0], torch.tensor(ref_proof_p, dtype=gpu.dtype))

    weights_in = gpu._gather_query_col(target)
    ghost = ~gpu.concrete
    positive = torch.clamp(weights_in, min=0.0) * ghost.to(gpu.dtype)
    env_conj_mass = positive.sum(dim=-1)
    env_conj_p = 1.0 - (1.0 + env_conj_mass).pow(-float(gpu.cfg.eta))
    ref_conj = ConjectureKernel(gpu.cfg.eta)
    ref_conj_p = ref_conj.success_probability(library, weights_in[0, 0].numpy())
    torch.testing.assert_close(env_conj_p[0, 0], torch.tensor(ref_conj_p, dtype=gpu.dtype))

    ghosts, probs = ref_conj.proposal_probabilities(library, weights_in[0, 0].numpy())
    env_probs = (positive[0, 0] / env_conj_mass[0, 0]).numpy()
    positive_ref = [(phi, prob) for phi, prob in zip(ghosts, probs) if prob > 0.0]
    assert [phi for phi, _ in positive_ref] == [4, 5]
    np.testing.assert_allclose(
        env_probs[[phi for phi, _ in positive_ref]],
        [prob for _, prob in positive_ref],
        rtol=1e-6,
        atol=1e-6,
    )

    query = StaticQueryModel(
        F_size=gpu.F,
        horizon_H=gpu.cfg.horizon_H,
        truth_prob=gpu.query_truth[0, 0].numpy(),
        weights=gpu.graph_weights[0].numpy(),
        rho=gpu.cfg.rho,
    )
    query.update(library)
    ref_prob, ref_time = query.prob_and_time(3)
    active = torch.zeros((gpu.B, gpu.A), dtype=torch.bool)
    active[0, 0] = True
    gpu._run_query_prob_time(active, target)
    torch.testing.assert_close(gpu.query_prob[0, 0, 3], torch.tensor(ref_prob, dtype=gpu.dtype))
    torch.testing.assert_close(gpu.query_time[0, 0, 3], torch.tensor(ref_time, dtype=gpu.dtype))


def test_observation_is_fixed_shape_and_on_device():
    gpu = _batch(n=4, seed=40)
    actor_obs = gpu.actor_obs()
    BA = gpu.B * gpu.A

    assert actor_obs["formula_features"].shape[0] == BA
    assert actor_obs["formula_features"].shape[1] == gpu.F
    assert actor_obs["formula_mask"].shape == (BA, gpu.F)
    assert actor_obs["formula_features"].device.type == "cpu"


def test_pre_resolved_theorems_are_prefix_and_targets_stay_random_suffix():
    gpu = _batch(
        n=8,
        seed=41,
        prob_initially_resolved=0.25,
        prob_initially_target=0.25,
    )
    resolved_theorem = gpu.public_resolved[:, :gpu.N] | gpu.public_resolved[:, gpu.N:]
    target_theorem = gpu.is_target[:, :gpu.N].bool() | gpu.is_target[:, gpu.N:].bool()
    expected_resolved = torch.zeros((gpu.B, gpu.N), dtype=torch.bool)
    expected_resolved[:, :3] = True

    assert torch.equal(resolved_theorem.cpu(), expected_resolved)
    assert torch.equal((resolved_theorem & target_theorem), torch.zeros_like(resolved_theorem))
    assert torch.equal(target_theorem[:, :3], torch.zeros((gpu.B, 3), dtype=torch.bool))
    assert torch.equal(target_theorem.sum(dim=1), torch.full((gpu.B,), 3, dtype=torch.long))


def test_obs_formula_cap_shrinks_obs_to_k():
    K = 6
    gpu = _batch(n=4, seed=40, obs_formula_cap=K)
    BA = gpu.B * gpu.A
    sel = gpu.select_topk_formulas()
    assert sel.shape == (gpu.B, gpu.A, K)

    actor_obs = gpu.actor_obs(sel)
    assert actor_obs["formula_features"].shape[:2] == (BA, K)
    assert actor_obs["query_related_edges"].shape == (BA, K, K)

    masks = gpu.available_action_masks(sel)
    assert masks["prove"].shape == (gpu.B, gpu.A, K)
    # Every selected slot id is concrete-or-ghost but always a valid formula id.
    assert int(sel.min()) >= 0 and int(sel.max()) < gpu.F


def test_obs_formula_cap_uses_context_not_target_held_market_query_priority():
    K = 4
    gpu = _batch(n=3, seed=77, obs_formula_cap=K)

    gpu.context_theorems[0, 0] = torch.tensor([0, 1])
    phi = 2
    gpu.concrete[0, 0, [phi, phi + gpu.N]] = True
    gpu.is_target[0, [phi, phi + gpu.N]] = 1.0
    gpu.s[0, phi, 0, 0] = 1.0
    gpu.market_active[0, phi] = True
    gpu.query_prob_known[0, 0, phi] = True

    sel = gpu.select_topk_formulas()
    assert sel[0, 0].tolist() == [0, 1, gpu.N, gpu.N + 1]
    assert phi not in sel[0, 0].tolist()


def test_formula_context_evicts_least_recently_touched_theorem():
    gpu = _batch(n=1, seed=88, obs_formula_cap=4)
    gpu.context_theorems[0, 0] = torch.tensor([0, 1])
    gpu.context_last_touched[0, 0] = torch.tensor([3, 7])
    gpu.timestep[0] = 9

    gpu._touch_context_theorems(
        torch.tensor([[2]], device=gpu.device),
        torch.tensor([[True]], device=gpu.device),
    )

    assert gpu.context_theorems[0, 0].tolist() == [2, 1]
    assert gpu.context_last_touched[0, 0].tolist() == [9, 7]


def test_mechanism_env_marks_agent_that_first_made_theorem_public(fake_mechanism_cls):
    gpu = _batch(n=1, seed=87)
    target = 2
    neg = target + gpu.N
    gpu.public_concrete[0, [target, neg]] = False
    gpu.concrete[0, :, [target, neg]] = False
    gpu.concrete[0, 1, target] = True
    gpu.concrete[0, 1, neg] = True
    gpu.market_active[:] = False
    seen = {}

    def update(s, a, e):
        seen["e"] = e.clone()
        return s

    gpu.mechanism = fake_mechanism_cls(update)
    zeros_ba = torch.zeros((gpu.B, gpu.A), dtype=torch.long)
    actions = TensorActions(
        math_type=zeros_ba - 1,
        math_formula=zeros_ba - 1,
        budget=zeros_ba.to(gpu.dtype),
        math_mode=zeros_ba,
        publish_statement=torch.zeros((gpu.B, gpu.A, gpu.F), dtype=torch.bool),
        publish_proof=torch.zeros((gpu.B, gpu.A, gpu.F), dtype=torch.bool),
        market_action=torch.zeros((gpu.B, gpu.A, gpu.F, gpu.dims.d_action), dtype=gpu.dtype),
    )
    actions.publish_statement[0, 1, target] = True

    proven_this_step, prover_onehot = gpu._process_publications(actions)
    market_action = torch.zeros((gpu.B, gpu.A, gpu.F, gpu.dims.d_action), dtype=gpu.dtype)
    gpu._advance_markets(market_action, proven_this_step, prover_onehot)

    e = seen["e"].reshape(gpu.B, gpu.F, gpu.A, gpu.dims.d_e)
    assert e[0, target, :, 5].tolist() == [0.0, 1.0]
    assert e[0, neg, :, 5].tolist() == [0.0, 1.0]


def test_actor_observation_content_matches_schema_columns():
    gpu = _batch(n=1, seed=101)
    phi = 3
    gpu.timestep[0] = 7
    gpu.public_concrete[0, phi] = True
    gpu.public_resolved[0, phi] = False
    gpu.is_target[0, phi] = 1.0
    gpu.job_type[0, 0] = gpu.JOB_PROVE
    gpu.job_target[0, 0] = phi
    gpu.job_tau[0, 0] = 5
    gpu.query_prob_known[0, 0, phi] = True
    gpu.query_prob[0, 0, phi] = 0.75

    obs = gpu.actor_obs()
    row = obs["formula_features"][0, phi]
    assert row[PUBLIC_CONCRETE].item() == 1.0
    assert row[PUBLIC_PROVEN].item() == 0.0
    assert row[FORMULA_IS_TARGET].item() == 1.0
    assert row[JOB_ACTIVE_TARGET].item() == 1.0
    assert row[JOB_TYPE_PROVE].item() == 1.0
    assert abs(row[JOB_TAU_REMAINING].item() - 5.0 / float(gpu.cfg.max_timestep)) < 1e-6
    assert row[QUERY_PROB_TIME_KNOWN].item() == 1.0
    assert abs(row[QUERY_PROBABILITY].item() - 0.75) < 1e-6
    assert abs(obs["agent_scalars"][0, AGENT_TIMESTEP].item() - 7.0 / float(gpu.cfg.max_timestep)) < 1e-6
    assert obs["agent_scalars"][0, AGENT_LAST_TURN].item() == 0.0


def test_query_related_references_enter_formula_context():
    gpu = _batch(n=1, seed=89, obs_formula_cap=4, query_num_related=1)
    target = 0
    related = 2
    gpu.context_theorems[0, 0] = torch.tensor([target, 1])
    gpu.context_last_touched[0, 0] = torch.tensor([5, 1])
    gpu.timestep[0] = 6
    gpu.concrete[0, 0, [target, target + gpu.N, related, related + gpu.N]] = True

    edge = (gpu._qw_src == target) & (gpu._qw_dst == related)
    if not bool(edge.any()):
        gpu._qw_src = torch.cat([gpu._qw_src, torch.tensor([target], device=gpu.device)])
        gpu._qw_dst = torch.cat([gpu._qw_dst, torch.tensor([related], device=gpu.device)])
        pad = torch.zeros((gpu.B, gpu.A, 1), dtype=gpu.dtype, device=gpu.device)
        gpu.query_values = torch.cat([gpu.query_values, pad], dim=2)
        edge = (gpu._qw_src == target) & (gpu._qw_dst == related)
    gpu.query_values[:, :, :] = 0.0
    gpu.query_values[0, 0, edge.nonzero().flatten()[0]] = 1.0

    gpu._touch_context_theorems(
        torch.tensor([[target]], device=gpu.device),
        torch.tensor([[True]], device=gpu.device),
    )
    gpu._run_query_related(
        torch.tensor([[True]], device=gpu.device),
        torch.tensor([[target]], device=gpu.device),
        torch.tensor([[0]], device=gpu.device),
    )

    assert related in gpu.context_theorems[0, 0].tolist()


def _force_query_edge(gpu, src, dst, value):
    edge = (gpu._qw_src == src) & (gpu._qw_dst == dst)
    if not bool(edge.any()):
        gpu._qw_src = torch.cat([gpu._qw_src, torch.tensor([src], device=gpu.device)])
        gpu._qw_dst = torch.cat([gpu._qw_dst, torch.tensor([dst], device=gpu.device)])
        pad = torch.zeros((gpu.B, gpu.A, 1), dtype=gpu.dtype, device=gpu.device)
        gpu.query_values = torch.cat([gpu.query_values, pad], dim=2)
        edge = (gpu._qw_src == src) & (gpu._qw_dst == dst)
    gpu.query_values[:, :, :] = 0.0
    gpu.query_values[0, 0, edge.nonzero().flatten()[0]] = value


def test_successful_outward_conjecture_records_query_related_edge():
    gpu = _batch(n=1, seed=90, query_num_related=0, eta=1000.0)
    target = 0
    proposal = 2
    gpu.concrete[0, 0, [target, target + gpu.N]] = True
    gpu.concrete[0, 0, [proposal, proposal + gpu.N]] = False
    _force_query_edge(gpu, target, proposal, 0.75)

    gpu.job_mode[0, 0] = 0
    gpu._advance_conj_jobs(
        torch.tensor([[True]], device=gpu.device),
        torch.tensor([[target]], device=gpu.device),
    )

    obs = gpu.actor_obs()
    row = 0 * gpu.A + 0
    assert bool(gpu.concrete[0, 0, proposal])
    assert torch.isclose(obs["query_related_edges"][row, target, proposal], torch.tensor(0.75))
    assert obs["query_related_edges"][row, proposal, target] == 0


def test_successful_inward_conjecture_records_query_related_edge():
    gpu = _batch(n=1, seed=91, query_num_related=0, eta=1000.0)
    target = 0
    proposal = 2
    gpu.concrete[0, 0, [target, target + gpu.N]] = True
    gpu.concrete[0, 0, [proposal, proposal + gpu.N]] = False
    _force_query_edge(gpu, proposal, target, 0.5)

    gpu.job_mode[0, 0] = 1
    gpu._advance_conj_jobs(
        torch.tensor([[True]], device=gpu.device),
        torch.tensor([[target]], device=gpu.device),
    )

    obs = gpu.actor_obs()
    row = 0 * gpu.A + 0
    assert bool(gpu.concrete[0, 0, proposal])
    assert torch.isclose(obs["query_related_edges"][row, proposal, target], torch.tensor(0.5))
    assert obs["query_related_edges"][row, target, proposal] == 0


def test_obs_formula_cap_off_matches_uncapped():
    gpu = _batch(n=3, seed=12)
    assert gpu.select_topk_formulas() is None
    a = gpu.actor_obs(None)
    a2 = gpu.actor_obs()
    assert torch.equal(a["formula_features"], a2["formula_features"])
    assert a["formula_features"].shape[1] == gpu.F


def test_collaborative_mode_uses_resident_mechanism_path():
    cfg = get_config()
    vcfg = build_env_config(cfg, level=cfg.levels[0])
    vcfg.control_mode = "collaborative"
    vcfg.first_prover_bonus = 0.5
    gpu = BatchedMathEnv.from_config(vcfg, batch_size=2, device=torch.device("cpu"), seeds=[90, 91])
    assert gpu.actor_obs()["formula_features"].shape[0] == gpu.B * gpu.A
