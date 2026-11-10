"""Write the board, action table and trained agents to web/data for the static site."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .board import COORDS, EDGES, KOMIKELU_START, TREWAS_START
from .game import ACTIONS, DEFAULT_RULES, ROLE_NAMES
from .policy import NumpyPolicy


def board_data() -> dict:
    return {
        "coords": COORDS,
        "edges": EDGES,
        "actions": [list(a) for a in ACTIONS],
        "komikeluStart": KOMIKELU_START,
        "trewasStart": TREWAS_START,
        "rules": {"capturesToWin": DEFAULT_RULES.captures_to_win, "maxPlies": DEFAULT_RULES.max_plies},
    }


def policy_to_web(policy: NumpyPolicy) -> tuple[bytes, list[list[int]]]:
    """Little-endian float32 values (w0, b0, w1, b1, ...) and each weight matrix's [rows, cols]."""
    blob = b"".join(
        np.asarray(a, dtype="<f4").tobytes() for w, b in zip(policy.weights, policy.biases) for a in (w, b)
    )
    return blob, [list(w.shape) for w in policy.weights]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("models"), help="folder with komikelu.npz and trewas.npz")
    parser.add_argument("--out", type=Path, default=Path("web/data"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "board.json").write_text(json.dumps(board_data(), separators=(",", ":")))

    agents = {}
    for name in ROLE_NAMES.values():
        source = args.models / f"{name}.npz"
        if not source.exists():
            print(f"skipping {name}: {source} not found")
            continue
        blob, shapes = policy_to_web(NumpyPolicy.load(source))
        (args.out / f"{name}.bin").write_bytes(blob)
        # The id ties recorded games to the exact agent they were played against.
        agents[name] = {"file": f"{name}.bin", "shapes": shapes, "id": hashlib.sha256(blob).hexdigest()[:12]}
        print(f"{name}: {len(blob) / 1024:.0f} KB")
    (args.out / "agents.json").write_text(json.dumps(agents, indent=1))


if __name__ == "__main__":
    main()
