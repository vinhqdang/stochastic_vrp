"""EXPERIMENT C stage definitions and arm plumbing (EXPERIMENT_C.md 4, 1.3, 1.7, 1.8, 4.3;
module MC4).

``c_stage_defs()`` returns the ordered stage table as plain data. Each entry carries the
keys of the spec: ``world`` (W0, W3, W6, W9, WS), ``arms`` (neural arm names), ``labels``
(ladder labels, () for none), ``feature_set`` (F0..F3, F0n, or None), ``routes``
("eligible", "subset" or "pilot"), ``reps``, ``N`` (100,000), ``arm_cfg`` (pairs as in B),
``eval_only`` and ``reuse_stage`` (the B stage whose models are evaluated, or None),
``dev_only``. ``to_stage_kwargs`` turns an entry into ``expa.sweep.StageSpec`` keyword
arguments; ``expa.sweep`` registers them.

Naming. The world-suffix spelling is lowercase: ``c_ladder_w0``, ``c_thr_w3``, ``c_boost_F0_w6``,
``c_rlsm_F3_ws`` (WS is spelled ``ws``, the same rule as the other worlds). Pilot stages
are the same entries prefixed ``c_pilot_`` (CON3-0 eligible routes, rep 0; EXPERIMENT_C 8).

Configurations. RLSM uses B's chosen configuration (``expb.stages.CHOSEN_ARM_CFG``). PPO-long
uses B's chosen PPO configuration with the pass count replaced by 9 (1.8). DOS reads
``results/expC/dos_choice.json`` written by ``python -m transvrp.expb.choose --dos-out``;
the stage refuses to fit when that record is missing. CDP takes ``CDPConfig``; CDP-true
sets ``rho_override = 0.6``.

Reused models (4.3). ``c_full_w6`` and ``c_ppob_w6`` have no fit. Their models are B's
``results/expB/models/<reuse_stage>`` files, whose fit commit is ``REUSE_FIT_COMMIT``
(bf3edab). ``check_reuse_models`` verifies every model file against B's manifest,
``check_reuse_gate`` covers the test_c gate through ``expc.gates.verify_reuse_stage``.
"""
from __future__ import annotations

import hashlib
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from ..expa.paths import Paths, RouteSpec, route_pool
from ..expa.train import ConstantPredictor
from ..expb import codecs
from ..expb.choose import load_chosen
from ..expb.registry import ARM_SPECS, fit_arm
from ..expb.stages import CHOSEN_ARM_CFG, LADDER_B, TUNE_GRID
from .boost_sets import BoostSetPredictor, fit_boost_set
from .cdp import CDPPredictor, CDPTables, N_EPS, N_GRID, N_NODES, build_tables, fit_cdp
from .rlsm_sets import fit_rlsm_set

CODE_DIR = Path(__file__).resolve().parents[2]
RESULTS_C = CODE_DIR / "results" / "expC"
DOS_CHOICE_PATH = RESULTS_C / "dos_choice.json"

STAGE_N = 100_000                       # section 3: training size N = 10^5
FIT_REPS = (0, 1)                       # section 4: replicates 0-1
PPO_LONG_PASSES = 9                     # 1.8: 3.6 times the source's 2*10^5 episodes
REUSE_FIT_COMMIT = "bf3edabd9857644c28ac4d301ea82063950715ff"   # B's fit commit F (4.3)
WORLDS = ("W0", "W3", "W6", "W9", "WS")
WORLD_SUFFIX = {"W0": "w0", "W3": "w3", "W6": "w6", "W9": "w9", "WS": "ws"}
FEATURE_SETS_W6 = ("F0", "F0n", "F1", "F2", "F3")
FEATURE_SETS_OTHER = ("F0", "F3")
CDP_TRUE_RHO = 0.6
PILOT_PREFIX = "c_pilot_"


@dataclass(frozen=True)
class CDPConfig:
    """CDP configuration (1.7). The grids default to the pre-registered 25 nodes, 256 Sobol
    draws and 1,001 load points; tests override them through ``arm_cfg`` only."""
    rho_override: float | None = None
    n_nodes: int = N_NODES
    n_eps: int = N_EPS
    n_grid: int = N_GRID


@dataclass(frozen=True)
class NoConfig:
    """Arms without tunable settings (BOOST with A's M11 parameters)."""


C_CONFIG_CLASSES: dict[str, type] = {"cdp": CDPConfig, "boost": NoConfig}


def _pairs_with_passes(pairs: tuple, passes: int) -> tuple:
    return tuple(p for p in pairs if p[0] != "passes") + (("passes", passes),)


def load_dos_pairs(path: Path = DOS_CHOICE_PATH) -> tuple:
    """DOS configuration pairs of the committed DOS record; () when it is missing."""
    return load_chosen(path, TUNE_GRID).get("dos", ())


def _world_tag(world: str) -> str:
    if world not in WORLD_SUFFIX:
        raise ValueError(f"world must be one of {WORLDS}, got {world!r}")
    return WORLD_SUFFIX[world]


def _def(world: str, *, arms=(), labels=(), feature_set=None, routes="eligible",
         reps=FIT_REPS, arm_cfg=(), eval_only=False, reuse_stage=None, dev_only=False) -> dict:
    return dict(world=world, arms=tuple(arms), labels=tuple(labels), feature_set=feature_set,
                routes=routes, reps=tuple(reps), N=STAGE_N, arm_cfg=tuple(arm_cfg),
                eval_only=eval_only, reuse_stage=reuse_stage, dev_only=dev_only)


def c_stage_defs(chosen: dict | None = None, dos_pairs: tuple | None = None) -> dict[str, dict]:
    """The EXPERIMENT C stage table (section 4), in spec order. ``chosen`` defaults to
    B's chosen configurations and ``dos_pairs`` to the committed DOS record."""
    chosen = CHOSEN_ARM_CFG if chosen is None else chosen
    dos_pairs = load_dos_pairs() if dos_pairs is None else dos_pairs
    rlsm = (("rlsm", chosen["rlsm"]),) if "rlsm" in chosen else ()
    ppo = chosen.get("ppo", ())
    out: dict[str, dict] = {}

    def put(name: str, kw: dict) -> None:
        out[name] = _def(**kw)

    for w in WORLDS:
        t = _world_tag(w)
        put(f"c_ladder_{t}", dict(world=w, labels=LADDER_B))
        put(f"c_thr_{t}", dict(world=w, arms=("thr",), arm_cfg=(("thr", ()),)))
        put(f"c_cdp_{t}", dict(world=w, arms=("cdp",), arm_cfg=(("cdp", ()),)))
    put("c_cdptrue_w6", dict(world="W6", arms=("cdp",),
                             arm_cfg=(("cdp", (("rho_override", CDP_TRUE_RHO),)),)))
    for w in WORLDS:
        sets = FEATURE_SETS_W6 if w == "W6" else FEATURE_SETS_OTHER
        for fs in sets:
            t = _world_tag(w)
            put(f"c_boost_{fs}_{t}", dict(world=w, arms=("boost",), feature_set=fs,
                                          arm_cfg=(("boost", ()),)))
            put(f"c_rlsm_{fs}_{t}", dict(world=w, arms=("rlsm",),
                                         feature_set=fs, arm_cfg=rlsm))
    put("c_dos_w6", dict(world="W6", arms=("dos",), arm_cfg=(("dos", dos_pairs),) if dos_pairs else ()))
    put("c_ppolong_w6", dict(world="W6", arms=("ppo",), routes="subset", reps=(0,),
                             arm_cfg=(("ppo", _pairs_with_passes(ppo, PPO_LONG_PASSES)),)
                             if ppo else ()))
    put("c_full_w6", dict(world="W6", arms=("full",), eval_only=True, reuse_stage="b_full"))
    put("c_ppob_w6", dict(world="W6", arms=("ppo",), routes="subset", reps=(0,),
                          eval_only=True, reuse_stage="b_ppo"))
    return out


def c_pilot_defs(defs: dict[str, dict] | None = None) -> dict[str, dict]:
    """Pilot (section 8, dev only): every fitted C stage except PPO-long, on the CON3-0
    eligible routes, rep 0, at N = 10^5. The eval-only stages are not pilot stages."""
    defs = c_stage_defs() if defs is None else defs
    out = {}
    for name, d in defs.items():
        if d["eval_only"] or name == "c_ppolong_w6":
            continue
        out[PILOT_PREFIX + name[len("c_"):]] = dict(d, routes="pilot", reps=(0,), dev_only=True)
    return out


def to_stage_kwargs(d: dict) -> dict:
    """``expa.sweep.StageSpec`` keyword arguments of one entry of ``c_stage_defs``."""
    N = d["N"]
    kw = dict(reps=d["reps"], routes=d["routes"], world=d["world"],
              feature_set=d["feature_set"] or "", family="C", arm_cfg=d["arm_cfg"],
              eval_only=d["eval_only"], neural=tuple((a, N) for a in d["arms"]),
              test_stream="test_c", test_allowed=not d["dev_only"])
    if d["labels"]:
        kw.update(ladder_N=(N,), ladder_labels=d["labels"])
    if d["reuse_stage"]:
        kw["reuse_models"] = f"expB/models/{d['reuse_stage']}"
    return kw


def feature_set_label(feature_set: str) -> str:
    """Column value of ``feature_set`` in merged rows ("none" when the stage has no set)."""
    return feature_set or "none"


# ---------------------------------------------------------------- arm configurations

def c_arm_config(sspec, arm: str):
    """Configuration object of one arm of a C stage (``StageSpec.arm_cfg`` pairs)."""
    entries = [pairs for a, pairs in sspec.arm_cfg if a == arm]
    if len(entries) != 1:
        hint = (" (DOS needs results/expC/dos_choice.json: python -m transvrp.expb.choose "
                "--dos-out results/expC/dos_choice.json)") if arm == "dos" else ""
        raise ValueError(f"stage has no configuration for arm {arm!r}{hint}")
    cls = C_CONFIG_CLASSES[arm] if arm in C_CONFIG_CLASSES else ARM_SPECS[arm].config_cls
    return cls(**dict(entries[0]))


# ---------------------------------------------------------------- fit

def _cdp_seed(spec: RouteSpec, rep: int) -> int:
    """Default seed of ``cdp.fit_cdp`` (crc32 of instance, route and replicate)."""
    return zlib.crc32(f"cdp|{spec.inst}|{spec.route_id}|{int(rep)}".encode("utf-8"))


def _boost_blob(models: dict, set_name: str, E0: float) -> dict:
    states = {}
    for k, preds in models.items():
        items = []
        for p in preds:
            if isinstance(p, BoostSetPredictor):
                items.append(("boost", p.model))
            elif isinstance(p, ConstantPredictor):
                items.append(("const", float(p.value)))
            else:
                raise TypeError(f"cannot pack predictor {type(p).__name__}")
        states[k] = items
    return {"family": "C", "arm": "boost", "set": set_name, "E0": float(E0), "states": states}


def c_unpack(blob: dict) -> dict:
    """Predictors of a C-family payload blob (BOOST only; CDP needs the training paths, see
    ``cdp_predictors``)."""
    if blob["arm"] != "boost":
        raise ValueError(f"{blob['arm']!r} payloads are unpacked by cdp_predictors")
    out: dict[int, list] = {}
    for k, items in blob["states"].items():
        out[k] = [BoostSetPredictor(v, blob["set"]) if kind == "boost"
                  else ConstantPredictor(v) for kind, v in items]
    return out


def cdp_train_paths(job, world: str, cache_dir: Path | None = None) -> Paths:
    """Training paths of a CDP job (N rows, the job's replicate, the stage's world): the
    same pool as the fit's. Every consumer of a CDP payload calls this."""
    return route_pool(job.inst, "train", job.rep, job.N, cache_dir=cache_dir,
                      routes=[job.route_id], world=world)[job.route_id]


def cdp_tables_of(blob: dict, spec: RouteSpec, train: Paths) -> CDPTables:
    """CDP tables of a payload, rebuilt from the stored model, the training paths of the fit
    and the stored grid settings. Payloads do not store the tables (they are large); the
    eval, the node gate and the tests all rebuild them here, so there is one source of
    truth. The rebuild is deterministic (same seed, same arrays)."""
    params = blob["params"]
    return build_tables(spec, blob["cdp_model"], train, params["n_nodes"],
                        params["n_eps"], params["n_grid"], params["seed"])


def cdp_predictors(blob: dict, spec: RouteSpec, train: Paths) -> dict:
    """Predictors of a CDP payload (tables from ``cdp_tables_of``)."""
    pred = CDPPredictor(blob["cdp_model"], cdp_tables_of(blob, spec, train))
    return {k: [pred] for k in range(1, spec.m)}


def _cdp_payload(spec: RouteSpec, paths: Paths, job, cfg: CDPConfig) -> dict:
    seed = _cdp_seed(spec, job.rep)
    res = fit_cdp(spec, paths, job.N, job.rep, rho_override=cfg.rho_override,
                  n_nodes=cfg.n_nodes, n_eps=cfg.n_eps, n_grid=cfg.n_grid, seed=seed)
    blob = {"family": "C", "arm": "cdp", "cdp_model": res["cdp_model"],
            "params": dict(rho_override=cfg.rho_override, n_nodes=cfg.n_nodes,
                           n_eps=cfg.n_eps, n_grid=cfg.n_grid, seed=seed)}
    return {"kind": "neural", "models": blob, "stats": res["stats"], "diag": res["diag"],
            "budget": res["budget"]}


def c_fit_payload(job, sspec, spec: RouteSpec, paths: Paths) -> dict:
    """Payload fields (models, stats, diag, budget) of one C neural job. ``job.arm`` is one
    of thr, dos, ppo (B's fitters), rlsm (B's fitter, or the set fitter when the stage has a
    feature set), boost (set fitter) and cdp."""
    cfg = c_arm_config(sspec, job.arm)
    if job.arm == "cdp":
        return _cdp_payload(spec, paths, job, cfg)
    if job.arm == "boost":
        t0 = time.perf_counter()
        models, stats, diag = fit_boost_set(spec, paths, job.N, job.rep, sspec.feature_set)
        budget = {"train_seconds": time.perf_counter() - t0, "feature_set": sspec.feature_set}
        return {"kind": "neural", "models": _boost_blob(models, sspec.feature_set, spec.E0),
                "stats": stats, "diag": diag, "budget": budget}
    if job.arm == "rlsm" and sspec.feature_set:
        models, stats, diag, budget = fit_rlsm_set(spec, paths, job.N, job.rep, cfg,
                                                   sspec.feature_set)
    else:
        models, stats, diag, budget = fit_arm(job.arm, spec, paths, job.N, job.rep, cfg)
    return {"kind": "neural", "models": codecs.pack_models(models, job.arm, spec.E0),
            "stats": stats, "diag": diag, "budget": budget}


# ---------------------------------------------------------------- reused models (4.3)

def _manifest_path(results_dir: Path, sspec) -> Path:
    b_stage = sspec.reuse_models.rsplit("/", 1)[-1]
    return Path(results_dir).parent / "expB" / b_stage / "models.sha256"


def read_manifest(path: Path) -> dict:
    """{file name: (sha256, fit commit, fit_tree_clean)} of a B models.sha256."""
    if not Path(path).exists():
        raise ValueError(f"reused stage needs B's manifest {path}")
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sha, name, commit, clean = line.split()
        out[name] = (sha, commit, clean)
    return out


def check_reuse_models(sspec, results_dir: Path, model_files: list[Path]) -> None:
    """Every model file to be evaluated hashes to B's manifest, is listed there, and was
    fitted at REUSE_FIT_COMMIT on a clean tree. Raises ValueError otherwise (before any
    data is generated)."""
    manifest = read_manifest(_manifest_path(results_dir, sspec))
    for path in model_files:
        name = path.name
        if name not in manifest:
            raise ValueError(f"reused model {name} is not in B's manifest")
        sha, commit, clean = manifest[name]
        if commit != REUSE_FIT_COMMIT or clean != "True":
            raise ValueError(f"reused model {name} has fit commit {commit} / clean {clean}, "
                             f"expected {REUSE_FIT_COMMIT} / True")
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if digest != sha:
            raise ValueError(f"reused model {name} does not match B's manifest "
                             f"(sha256 {digest} != {sha})")


def check_reuse_gate(stage: str, eval_set: str, frozen: str | None) -> None:
    """test_c evaluation of a reused stage: the frozen commit must be B's fit commit, and
    ``expc.gates.verify_reuse_stage`` must accept the stage (4.3 items a to c). Until that
    function exists the test_c path is refused."""
    if eval_set != "test_c":
        return
    if frozen != REUSE_FIT_COMMIT:
        raise ValueError(f"reused stage {stage!r} needs the frozen commit {REUSE_FIT_COMMIT}")
    try:
        from . import gates      # noqa: PLC0415 (written separately; may not exist yet)
    except ImportError:
        gates = None
    verify = getattr(gates, "verify_reuse_stage", None)
    if verify is None:
        raise NotImplementedError("TODO: expc.gates.verify_reuse_stage(stage) is not "
                                  "available; test_c evaluation of reused stages is refused")
    verify(stage)
