"""Mutation tests for the move kernels: each deliberately broken variant of moves.py must be
caught by the named tests.  Runs in a temporary copy, the working tree is never modified.
Usage: python tools/mutation_check.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# control: the unmodified code must pass in the same harness, otherwise a "caught" mutation
# could just be an environment failure
MUTATIONS = [
    ("CONTROL (no mutation, must pass)", "np.exp(-beta * de) * q_old / q_new:",
     "np.exp(-beta * de) * q_old / q_new:", "polyspin/tests/test_followup1.py -k four_spin"),
    ("overlap acceptance without proposal-density ratio",
     "np.exp(-beta * de) * q_old / q_new:", "np.exp(-beta * de):",
     "polyspin/tests/test_m1.py -k detailed_balance"),
    ("mixture density: only one vMF component",
     """    for j in range(n):
        if j != i:
            x = vx * spins[a, j, 0] + vy * spins[a, j, 1] + vz * spins[a, j, 2]
            acc += _vmf_density(x, kappa)
    return 0.5 / FOUR_PI + 0.5 * acc / (n - 1)""",
     """    j0 = 0 if i != 0 else 1
    x = vx * spins[a, j0, 0] + vy * spins[a, j0, 1] + vz * spins[a, j0, 2]
    acc = (n - 1) * _vmf_density(x, kappa)
    return 0.5 / FOUR_PI + 0.5 * acc / (n - 1)""",
     "polyspin/tests/test_followup1.py -k four_spin"),
    ("centre-selection weight 1/n instead of 1/(n-1)",
     "0.5 * acc / (n - 1)", "0.5 * acc / n",
     "polyspin/tests/test_followup1.py -k four_spin"),
    ("replica exchange: swapped sign of the energy difference",
     "d = (betas[t] - betas[t + 1]) * (energies_by_T[t] - energies_by_T[t + 1])",
     "d = (betas[t] - betas[t + 1]) * (energies_by_T[t + 1] - energies_by_T[t])",
     "polyspin/tests/test_pt.py -k detailed", "pt.py"),
]


def main():
    ok = True
    for entry in MUTATIONS:
        name, old, new, tests = entry[:4]
        target = entry[4] if len(entry) > 4 else "moves.py"
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(ROOT / "polyspin", Path(tmp) / "polyspin",
                            ignore=shutil.ignore_patterns("__pycache__", "*.nbi", "*.nbc"))
            f = Path(tmp) / "polyspin" / target
            src = f.read_text()
            assert old in src, f"pattern for '{name}' not found"
            f.write_text(src.replace(old, new))
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"]
                               + tests.replace("polyspin/", str(Path(tmp) / "polyspin") + "/", 1).split(),
                               cwd=tmp, env={"PYTHONPATH": tmp, "PATH": "/usr/bin:/bin:/usr/local/bin"},
                               capture_output=True, text=True)
        if name.startswith("CONTROL"):
            ok &= r.returncode == 0
            print(f"{'PASS   ' if r.returncode == 0 else 'FAIL   '} {name}")
            if r.returncode:
                print(r.stdout[-800:], r.stderr[-800:])
            continue
        caught = r.returncode != 0
        ok &= caught
        print(f"{'CAUGHT ' if caught else 'MISSED '} {name}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
