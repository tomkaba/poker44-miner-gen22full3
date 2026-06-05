#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker44.gen20_hybrid_model import Gen20HybridV1
from poker44.miner_heuristics import get_chunk_scorer_startup_check, score_chunk_runtime_with_route


ACTION_MAP = {
    "fold": 1,
    "call": 2,
    "raise": 3,
    "check": 4,
    "bet": 5,
    "all_in": 6,
}

STREET_MAP = {
    "preflop": 1,
    "flop": 2,
    "turn": 3,
    "river": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare standalone gen22 vote101 evaluator with miner runtime output")
    parser.add_argument("--log", default="/home/tk/training_gen22/miner_221.log")
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "weights" / "gen22_hybrid_full_0.0595.pt"))
    parser.add_argument("--minichunk-threshold", type=float, default=0.0595)
    parser.add_argument("--votes-per-parent", type=int, default=101)
    parser.add_argument("--positive-votes-required", type=int, default=66)
    parser.add_argument("--min-hands", type=int, default=4)
    parser.add_argument("--runtime-seed", type=int, default=20260605)
    parser.add_argument("--task-limit", type=int, default=2)
    parser.add_argument("--chunks-per-task", type=int, default=10)
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _encode_chunk(chunk: list[dict[str, Any]], max_hands: int, max_actions: int) -> dict[str, np.ndarray]:
    shape = (max_hands, max_actions)
    arr_action_type = np.zeros(shape, dtype=np.int64)
    arr_street = np.zeros(shape, dtype=np.int64)
    arr_actor_seat = np.zeros(shape, dtype=np.int64)
    arr_amount = np.zeros(shape, dtype=np.float32)
    arr_raise_to = np.zeros(shape, dtype=np.float32)
    arr_call_to = np.zeros(shape, dtype=np.float32)
    arr_norm_bb = np.zeros(shape, dtype=np.float32)
    arr_pot_before = np.zeros(shape, dtype=np.float32)
    arr_pot_after = np.zeros(shape, dtype=np.float32)
    arr_raise_miss = np.zeros(shape, dtype=np.float32)
    arr_call_miss = np.zeros(shape, dtype=np.float32)
    arr_valid = np.zeros(shape, dtype=np.float32)

    for hand_index, hand in enumerate(chunk[:max_hands]):
        for action_index, action in enumerate((hand.get("actions") or [])[:max_actions]):
            action_type = str(action.get("action_type") or "").lower()
            street = str(action.get("street") or "").lower()
            seat = action.get("actor_seat")
            arr_action_type[hand_index, action_index] = ACTION_MAP.get(action_type, 0)
            arr_street[hand_index, action_index] = STREET_MAP.get(street, 0)
            arr_actor_seat[hand_index, action_index] = int(seat) + 1 if isinstance(seat, int) and seat >= 0 else 0
            arr_amount[hand_index, action_index] = _safe_float(action.get("amount"))
            raise_to = action.get("raise_to")
            call_to = action.get("call_to")
            arr_raise_miss[hand_index, action_index] = 1.0 if raise_to is None else 0.0
            arr_call_miss[hand_index, action_index] = 1.0 if call_to is None else 0.0
            arr_raise_to[hand_index, action_index] = _safe_float(raise_to)
            arr_call_to[hand_index, action_index] = _safe_float(call_to)
            arr_norm_bb[hand_index, action_index] = _safe_float(action.get("normalized_amount_bb"))
            arr_pot_before[hand_index, action_index] = _safe_float(action.get("pot_before"))
            arr_pot_after[hand_index, action_index] = _safe_float(action.get("pot_after"))
            arr_valid[hand_index, action_index] = 1.0

    return {
        "action_type": arr_action_type,
        "street": arr_street,
        "actor_seat": arr_actor_seat,
        "amount": arr_amount,
        "raise_to": arr_raise_to,
        "call_to": arr_call_to,
        "norm_amount_bb": arr_norm_bb,
        "pot_before": arr_pot_before,
        "pot_after": arr_pot_after,
        "raise_to_missing": arr_raise_miss,
        "call_to_missing": arr_call_miss,
        "valid_mask": arr_valid,
    }


def _sample_hand_indices(rng: random.Random, hand_count: int, sample_size: int) -> list[int]:
    if hand_count >= sample_size:
        return rng.sample(range(hand_count), sample_size)
    return [rng.randrange(hand_count) for _ in range(sample_size)]


def _chunk_seed(chunk: list[dict[str, Any]], runtime_seed: int) -> int:
    chunk_bytes = json.dumps(chunk, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.blake2b(chunk_bytes, digest_size=8, person=b"p44vote1").digest()
    return int.from_bytes(digest, "big") ^ int(runtime_seed)


def _load_chunk_features():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from poker44_ml.features import chunk_features  # type: ignore

    return chunk_features


def _load_model(checkpoint_path: Path) -> tuple[Gen20HybridV1, dict[str, Any]]:
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = raw_checkpoint.get("model_state", raw_checkpoint)
    checkpoint_args = dict(raw_checkpoint.get("args", {})) if isinstance(raw_checkpoint, dict) else {}
    model = Gen20HybridV1(dropout=float(checkpoint_args.get("dropout", 0.10)))
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint_args


def _standalone_score_chunk(
    chunk: list[dict[str, Any]],
    model: Gen20HybridV1,
    chunk_features,
    *,
    minichunk_threshold: float,
    votes_per_parent: int,
    positive_votes_required: int,
    min_hands: int,
    max_hands: int,
    max_actions: int,
    runtime_seed: int,
) -> dict[str, Any]:
    rng = random.Random(_chunk_seed(chunk, runtime_seed))
    encoded_rows = {key: [] for key in [
        "action_type",
        "street",
        "actor_seat",
        "amount",
        "raise_to",
        "call_to",
        "norm_amount_bb",
        "pot_before",
        "pot_after",
        "raise_to_missing",
        "call_to_missing",
        "valid_mask",
    ]}
    dense_names: list[str] | None = None
    dense_rows: list[np.ndarray] = []

    max_sample_hands = min(max_hands, len(chunk))
    min_sample_hands = min(min_hands, max_sample_hands)
    for _ in range(votes_per_parent):
        sample_size = rng.randint(min_sample_hands, max_sample_hands)
        hand_indices = _sample_hand_indices(rng, len(chunk), sample_size)
        minichunk = [chunk[index] for index in hand_indices]
        encoded = _encode_chunk(minichunk, max_hands=max_hands, max_actions=max_actions)
        feature_map = chunk_features(minichunk)
        if dense_names is None:
            dense_names = sorted(feature_map.keys())
        dense_rows.append(np.asarray([float(feature_map.get(name, 0.0)) for name in dense_names], dtype=np.float32))
        for key, value in encoded.items():
            encoded_rows[key].append(value)

    batch = {key: np.stack(rows, axis=0) for key, rows in encoded_rows.items()}
    batch["dense_features"] = np.stack(dense_rows, axis=0)
    tensors = []
    for key in [
        "action_type",
        "street",
        "actor_seat",
        "amount",
        "raise_to",
        "call_to",
        "norm_amount_bb",
        "pot_before",
        "pot_after",
        "raise_to_missing",
        "call_to_missing",
        "valid_mask",
    ]:
        tensor = torch.from_numpy(batch[key])
        tensors.append(tensor.long() if key in {"action_type", "street", "actor_seat"} else tensor.float())
    tensors.append(torch.from_numpy(batch["dense_features"]).float())
    with torch.no_grad():
        scores = model(*tensors).detach().cpu().numpy().astype(np.float32, copy=False)
    positive_votes = int(np.count_nonzero(scores >= minichunk_threshold))
    vote_fraction = positive_votes / max(votes_per_parent, 1)
    return {
        "score": float(round(vote_fraction, 6)),
        "prediction": bool(positive_votes >= positive_votes_required),
        "positive_votes": positive_votes,
    }


def main() -> int:
    args = parse_args()
    startup = get_chunk_scorer_startup_check("runtime")
    if not startup.get("ok"):
        raise SystemExit(f"Miner runtime startup check failed: {startup}")

    model: Gen20HybridV1 | None = None
    checkpoint_args: dict[str, Any] = {}
    chunk_features = None
    max_hands = 8
    max_actions = 32
    if not args.runtime_only:
        model, checkpoint_args = _load_model(Path(args.checkpoint).expanduser().resolve())
        max_hands = int(checkpoint_args.get("max_hands", 8))
        max_actions = int(checkpoint_args.get("max_actions", 32))
        chunk_features = _load_chunk_features()

    log_entries = [json.loads(line) for line in Path(args.log).expanduser().resolve().read_text().splitlines() if line.strip()]
    if args.task_limit > 0:
        log_entries = log_entries[: args.task_limit]

    results: list[dict[str, Any]] = []
    mismatch_count = 0
    total = 0
    decision_threshold = args.positive_votes_required / args.votes_per_parent

    for task_index, entry in enumerate(log_entries, start=1):
        chunks = entry.get("chunks") or []
        if args.chunks_per_task > 0:
            chunks = chunks[: args.chunks_per_task]
        task_match_count = 0
        task_mismatch_count = 0
        task_miner_responses: list[bool] = []
        print(f"[task {task_index}] chunks={len(chunks)} mode={'runtime-only' if args.runtime_only else 'parity'}", flush=True)
        miner_task_elapsed_sec = 0.0
        task_rows: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(chunks):
            miner_started_at = time.perf_counter()
            miner_score, miner_route = score_chunk_runtime_with_route(chunk)
            miner_chunk_elapsed_sec = time.perf_counter() - miner_started_at
            miner_task_elapsed_sec += miner_chunk_elapsed_sec
            miner_pred = bool(miner_score >= decision_threshold)
            task_miner_responses.append(miner_pred)
            if args.runtime_only:
                total += 1
                task_rows.append(
                    {
                        "chunk_index": chunk_index,
                        "chunk_runtime_s": round(miner_chunk_elapsed_sec, 6),
                        "miner_score": float(miner_score),
                        "miner_pred": miner_pred,
                        "miner_route": miner_route,
                    }
                )
                continue

            standalone = _standalone_score_chunk(
                chunk,
                model,
                chunk_features,
                minichunk_threshold=args.minichunk_threshold,
                votes_per_parent=args.votes_per_parent,
                positive_votes_required=args.positive_votes_required,
                min_hands=args.min_hands,
                max_hands=max_hands,
                max_actions=max_actions,
                runtime_seed=args.runtime_seed,
            )
            matches = abs(float(miner_score) - float(standalone["score"])) < 1e-6 and miner_pred == bool(standalone["prediction"])
            mismatch_count += 0 if matches else 1
            total += 1
            task_match_count += 1 if matches else 0
            task_mismatch_count += 0 if matches else 1
            print(
                "[task {task} chunk {chunk}] {status} "
                "standalone={standalone_score:.6f} ({standalone_votes}/{votes}) pred={standalone_pred} "
                "miner={miner_score:.6f} pred={miner_pred} route={route}".format(
                    task=task_index,
                    chunk=chunk_index,
                    status="MATCH" if matches else "MISMATCH",
                    standalone_score=float(standalone["score"]),
                    standalone_votes=int(standalone["positive_votes"]),
                    votes=args.votes_per_parent,
                    standalone_pred=str(bool(standalone["prediction"])).upper(),
                    miner_score=float(miner_score),
                    miner_pred=str(bool(miner_pred)).upper(),
                    route=miner_route,
                ),
                flush=True,
            )
            task_rows.append(
                {
                    "chunk_index": chunk_index,
                    "standalone_score": standalone["score"],
                    "standalone_positive_votes": standalone["positive_votes"],
                    "standalone_pred": standalone["prediction"],
                    "miner_score": float(miner_score),
                    "miner_pred": miner_pred,
                    "miner_route": miner_route,
                    "matches": matches,
                }
            )
        if args.runtime_only:
            miner_response_text = "[" + ", ".join(str(value).upper() for value in task_miner_responses) + "]"
            print(
                f"[task {task_index} runtime summary] chunks={len(task_rows)} miner_runtime_time_s={miner_task_elapsed_sec:.3f} miner_response={miner_response_text}",
                flush=True,
            )
            results.append(
                {
                    "task_index": task_index,
                    "miner_runtime_time_s": round(miner_task_elapsed_sec, 6),
                    "miner_response": task_miner_responses,
                    "chunks": task_rows,
                }
            )
            continue

        print(
            f"[task {task_index} summary] chunks={len(task_rows)} match={task_match_count} mismatch={task_mismatch_count} miner_model_time_s={miner_task_elapsed_sec:.3f}",
            flush=True,
        )
        results.append({"task_index": task_index, "chunks": task_rows})

    summary = {
        "log": str(Path(args.log).expanduser().resolve()),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "runtime_only": bool(args.runtime_only),
        "total_chunks": total,
        "mismatch_count": mismatch_count,
        "match_rate": (total - mismatch_count) / max(total, 1),
        "tasks": results,
    }

    if args.runtime_only:
        print(f"[runtime-only] total_chunks={total}")
    else:
        print(f"[parity] total_chunks={total} mismatch_count={mismatch_count} match_rate={summary['match_rate']:.4f}")
    out_path = Path(args.out).expanduser().resolve() if args.out else REPO_ROOT / "models" / "gen22full3_parity_check.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote={out_path}")
    return 0 if args.runtime_only or mismatch_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())