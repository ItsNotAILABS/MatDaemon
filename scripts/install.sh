#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install matdaemon

cat <<'MSG'
MatDaemon installed.

Try it now:
  python - <<'PY'
import numpy as np
import matdaemon as md
A = np.eye(4, dtype=np.float32)
B = np.ones((4, 4), dtype=np.float32)
print(md.matmul(A, B))
PY
MSG
