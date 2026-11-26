"""Switch Pyomo/IDAES between Ipopt builds (TODO D6).

IDAES puts its own bin directory (Windows build: Ipopt 3.13.2) first on
PATH. The DEF flowsheet (def_lowp_flowsheet.py) only converges with Ipopt
3.14, while the ammonia flowsheet's B6 initialization is tuned for (and
validated with) 3.13.2. So 3.14 is used only around DEF solves:

    with solver_path.ipopt314():
        ...solve the DEF model...

IPOPT_314_EXE overrides the default COIN-OR install location below.
"""
import contextlib
import os
from pathlib import Path

from pyomo.common import Executable

_DEFAULT = Path.home() / "ipopt-3.14" / "Ipopt-3.14.20-win64-msvs2022-md" \
    / "bin" / "ipopt.exe"


def exe_314():
    p = os.environ.get("IPOPT_314_EXE") or str(_DEFAULT)
    return p if Path(p).exists() else None


@contextlib.contextmanager
def ipopt314():
    exe = exe_314()
    if exe is None:          # not installed: fall back silently to IDAES'
        yield
        return
    ex = Executable("ipopt")
    old = ex.path()
    ex.set_path(exe)
    try:
        yield
    finally:
        ex.set_path(old)
