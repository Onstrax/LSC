"""Respuestas sintéticas rotuladas del contrato `lsc.http.v1`.

Cubren accepted / retry / error, incluidos fallos. Propósito: probar
integración y documentar el contrato, no medir reconocimiento. La demostración
final y los reportes científicos EXCLUYEN fixtures.
"""

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent

__all__ = ["FIXTURES_DIR"]
