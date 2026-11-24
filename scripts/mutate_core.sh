#!/usr/bin/env bash
# Mutation check for the verified core: every planted bug must make Verus
# fail. A surviving mutant means a proof is vacuous or a spec is too weak.
#   scripts/mutate_core.sh       (run scripts/verus.sh --setup first)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Use exactly the release scripts/verus.sh pins, never "whatever is cached".
VERUS_RELEASE="$(sed -n 's/^VERUS_RELEASE="\(.*\)"$/\1/p' "$ROOT/scripts/verus.sh")"
VERUS="$ROOT/.verus/$VERUS_RELEASE/verus-x86-linux/verus"
[ -x "$VERUS" ] || { echo "no verus at $VERUS: run scripts/verus.sh --setup"; exit 2; }
SRC="$ROOT/crates/minicut-core/src/lib.rs"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
survived=0

# Stability first: the unmutated core must verify standalone under several
# SMT seeds. A proof that only passes in one context or with one seed is
# brittle (found in practice: a division in an ensures clause verified only
# when another section followed it).
for seed in 0 1 2 3 4 5 6 7 8 9; do
  if ! "$VERUS" --crate-type=lib --smt-option smt.random_seed=$seed "$SRC" >/dev/null 2>&1; then
    echo "UNSTABLE: core fails with smt.random_seed=$seed"; survived=1
  fi
done
# A baseline that does not verify would make every mutant look "killed".
[ $survived = 0 ] || { echo "baseline does not verify cleanly; mutants not run"; exit 1; }
echo "stable:   core verifies under seeds 0-9"

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
mutant "speed meaning inverted (2x reported as longer)" \
  "Edit::Speed { factor_pm } => c.duration_ms * 1000 / factor_pm as int," \
  "Edit::Speed { factor_pm } => c.duration_ms * factor_pm as int / 1000,"
mutant "mute reported as dropping the picture's duration" \
  "        _ => c.duration_ms as int,
    }
}" \
  "        Edit::Mute => 0,
        _ => c.duration_ms as int,
    }
}"
mutant "trim admits a zero-length range" \
  "Edit::Trim { start_ms, end_ms } => start_ms < end_ms && end_ms <= c.duration_ms,
        Edit::CutOut { start_ms, end_ms } => start_ms < end_ms && end_ms <= c.duration_ms,
        Edit::Concat { n_inputs } => n_inputs >= 2,
        Edit::Scale { width, height } => width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0,
        Edit::Speed { factor_pm } => 250 <= factor_pm && factor_pm <= 4000," \
  "Edit::Trim { start_ms, end_ms } => start_ms <= end_ms && end_ms <= c.duration_ms,
        Edit::CutOut { start_ms, end_ms } => start_ms < end_ms && end_ms <= c.duration_ms,
        Edit::Concat { n_inputs } => n_inputs >= 2,
        Edit::Scale { width, height } => width > 0 && height > 0 && width % 2 == 0 && height % 2 == 0,
        Edit::Speed { factor_pm } => 250 <= factor_pm && factor_pm <= 4000,"
mutant "RemoveLast cuts from the wrong end" \
  "Intent::RemoveLast { ms } => Edit::CutOut { start_ms: c.duration_ms - ms, end_ms: c.duration_ms }," \
  "Intent::RemoveLast { ms } => Edit::CutOut { start_ms: 0, end_ms: ms },"

mutant "try_label skips the runtime validator" \
  "    if validate(e, c) { Some(e) } else { None }
}" \
  "    Some(e)
}"
mutant "pct_of truncates instead of refusing" "    if p % 100 == 0 {" "    if true {"
mutant "half size accepts a source whose half is odd" \
  "if w > 0 && h > 0 && w % 4 == 0 && h % 4 == 0 {" "if w > 0 && h > 0 && w % 2 == 0 && h % 2 == 0 {"
mutant "quarter size divides by 2" "let (rw, rh) = (w / 4, h / 4);" "let (rw, rh) = (w / 2, h / 2);"
mutant "'25% faster' read as +25 per-mille" "Some(1000 + 10 * p)" "Some(1000 + p)"
mutant "'at 80% speed' read as 80 per-mille" "Some(10 * p) }" "Some(p) }"
mutant "seconds text drops the fraction" \
  "    (ms / 1000, ms % 1000)
}" \
  "    (ms / 1000, 0)
}"
mutant "try_label forgets the KeepLast bound" \
  "    let ok = match i {
        Intent::KeepLast { ms } => ms <= c.duration_ms," \
  "    let ok = match i {
        Intent::KeepLast { ms } => true,"
mutant "kept_intervals drops cut_out's tail" \
  "let b = if end_ms < c.duration_ms { Some((end_ms, c.duration_ms)) } else { None };" \
  "let b: Option<(u64, u64)> = None;"
mutant "kept_intervals keeps part of the cut" "Some((0u64, start_ms))" "Some((0u64, start_ms + 1))"
mutant "ambiguity: other reading of 'first N' is 'last N'" \
  "        Intent::KeepFirst { ms } => Intent::RemoveFirst { ms },
        Intent::RemoveFirst { ms } => Intent::KeepFirst { ms },
        Intent::KeepLast { ms } => Intent::RemoveLast { ms },
        Intent::RemoveLast { ms } => Intent::KeepLast { ms },
    }
}

pub fn flip" \
  "        Intent::KeepFirst { ms } => Intent::RemoveLast { ms },
        Intent::RemoveFirst { ms } => Intent::KeepFirst { ms },
        Intent::KeepLast { ms } => Intent::RemoveLast { ms },
        Intent::RemoveLast { ms } => Intent::KeepLast { ms },
    }
}

pub fn flip"
mutant "ambiguity: flip returns the same reading" \
  "        Intent::KeepRange { start_ms, end_ms } => Intent::RemoveRange { start_ms, end_ms },
        Intent::RemoveRange { start_ms, end_ms } => Intent::KeepRange { start_ms, end_ms },
        Intent::KeepFirst { ms } => Intent::RemoveFirst { ms },
        Intent::RemoveFirst { ms } => Intent::KeepFirst { ms },
        Intent::KeepLast { ms } => Intent::RemoveLast { ms },
        Intent::RemoveLast { ms } => Intent::KeepLast { ms },
    }
}

pub proof fn" \
  "        Intent::KeepRange { start_ms, end_ms } => Intent::KeepRange { start_ms, end_ms },
        Intent::RemoveRange { start_ms, end_ms } => Intent::KeepRange { start_ms, end_ms },
        Intent::KeepFirst { ms } => Intent::RemoveFirst { ms },
        Intent::RemoveFirst { ms } => Intent::KeepFirst { ms },
        Intent::KeepLast { ms } => Intent::RemoveLast { ms },
        Intent::RemoveLast { ms } => Intent::KeepLast { ms },
    }
}

pub proof fn"

exit $survived
