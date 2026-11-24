#!/usr/bin/env bash
# Mutation check for the verified core: every planted bug must make Verus
# fail. A surviving mutant means a proof is vacuous or a spec is too weak.
#   scripts/mutate_core.sh       (run scripts/verus.sh --setup first)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERUS="$(ls -d "$ROOT"/.verus/*/verus-x86-linux | tail -1)/verus"
SRC="$ROOT/crates/minicut-core/src/lib.rs"
WORK="$(mktemp -d -p "${TMPDIR:-$ROOT/target}")"
trap 'rm -rf "$WORK"' EXIT
survived=0

# Stability first: the unmutated core must verify standalone under several
# SMT seeds. A proof that only passes in one context or with one seed is
# brittle (found in practice: a division in an ensures clause verified only
# when another section followed it).
for seed in 0 1 2 3 7; do
  if ! "$VERUS" --crate-type=lib --smt-option smt.random_seed=$seed "$SRC" >/dev/null 2>&1; then
    echo "UNSTABLE: core fails with smt.random_seed=$seed"; survived=1
  fi
done
[ $survived = 0 ] && echo "stable:   core verifies under seeds 0 1 2 3 7"

mutant() {  # mutant "<description>" "<exact old text>" "<new text>"
  python3 - "$SRC" "$WORK/m.rs" "$2" "$3" <<'EOF' || { echo "BAD PATTERN: $1"; survived=1; return; }
import sys
src, dst, old, new = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
text = open(src).read()
if old not in text:
    sys.exit(1)
open(dst, "w").write(text.replace(old, new, 1))
EOF
  if "$VERUS" --crate-type=lib "$WORK/m.rs" >/dev/null 2>&1; then
    echo "SURVIVED: $1"; survived=1
  else
    echo "killed:   $1"
  fi
}

mutant "validate forgets speed-1x no-op" \
  "Edit::Speed { factor_pm } => factor_pm != 1000,
        _ => true,
    }
}

// ----" \
  "Edit::Speed { factor_pm } => true,
        _ => true,
    }
}

// ----"
mutant "validate allows odd dims" \
  "Edit::Scale { width, height } => width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0,
        Edit::Speed { factor_pm } => 250 <= factor_pm && factor_pm <= 4000," \
  "Edit::Scale { width, height } => width > 0 && height > 0,
        Edit::Speed { factor_pm } => 250 <= factor_pm && factor_pm <= 4000,"
mutant "atempo wrong second stage" "(2000, Some(f / 2))" "(2000, Some(f / 3))"
mutant "atempo stage out of range" "(500, Some(f * 2))" "(250, Some(f * 4))"
mutant "clarify missing lists provided args" "let m = r & !provided;" "let m = r & provided;"
mutant "keep/remove inversion" \
  "Intent::KeepLast { ms } => Edit::Trim { start_ms: c.duration_ms - ms, end_ms: c.duration_ms }," \
  "Intent::KeepLast { ms } => Edit::CutOut { start_ms: c.duration_ms - ms, end_ms: c.duration_ms },"
mutant "off-by-one label" \
  "Intent::KeepFirst { ms } => Edit::Trim { start_ms: 0, end_ms: ms }," \
  "Intent::KeepFirst { ms } => Edit::Trim { start_ms: 0, end_ms: ms + 1 },"
mutant "clock drops the fraction (v1 dataset bug)" \
  "(ms / 60000, rem / 1000, rem % 1000)" "(ms / 60000, rem / 1000, 0)"

exit $survived
