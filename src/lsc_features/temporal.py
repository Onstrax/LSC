"""Muestreo temporal de índices (design §15).

``sample_indices(n)`` toma 6 índices de ``n`` cuadros efectivos (``n >= 6``) con
redondeo half-up determinista (NO el ``round()`` nativo de Python, que es
half-to-even). Garantiza ``idx[0] == 0`` e ``idx[5] == n - 1`` por construcción.

Tabla anclada (design §15):

    N=6  -> [0, 1, 2, 3, 4, 5]
    N=7  -> [0, 1, 2, 4, 5, 6]
    N=10 -> [0, 2, 4, 5, 7, 9]
    N=11 -> [0, 2, 4, 6, 8, 10]
"""

from __future__ import annotations

import math


def sample_indices(n: int) -> list[int]:
    """Seis índices temporales de ``n`` cuadros (half-up determinista, §15)."""
    assert n >= 6
    idx = [int(math.floor(k * (n - 1) / 5 + 0.5)) for k in range(6)]
    # Invariantes garantizadas por construcción.
    assert idx[0] == 0 and idx[5] == n - 1
    return idx
