"""
Justificación de las abstracciones (Principio III): `freeze` / `load_frozen` se usan en todas las
etapas (b) y mantienen el manifiesto entre ejecuciones (c); `TestLedger` guarda el estado de las
evaluaciones en test entre ejecuciones (c).

Todas las rutas son relativas al directorio de trabajo, que es la raíz del proyecto.
"""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

MANIFEST_PATH = Path("data/processed/manifest.json")
LEDGER_PATH = Path("data/processed/test_ledger.json")


def utc_now() -> datetime:
    """Hora actual en UTC.

    Retorno:
        datetime: instante actual con zona UTC.
    """
    return datetime.now(UTC)


def _iso(ts: datetime) -> str:
    """Formatea una marca de tiempo en ISO 8601 con microsegundos."""
    return ts.isoformat(timespec="microseconds")


def _parse(ts: str) -> datetime:
    """Lee una marca de tiempo ISO 8601."""
    return datetime.fromisoformat(ts)


def sha256_file(path: str | Path) -> str:
    """Calcula el SHA-256 de un archivo.

    Parámetros:
        path (str | Path): ruta del archivo.

    Retorno:
        str: hash hexadecimal.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_obj(obj: Any) -> str:
    """Calcula el SHA-256 de un objeto serializable en JSON (claves ordenadas).

    Parámetros:
        obj (Any): objeto JSON.

    Retorno:
        str: hash hexadecimal.
    """
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def load_manifest() -> dict:
    """Lee `data/processed/manifest.json`, o devuelve uno vacío si no existe.

    Retorno:
        dict: manifiesto con `version`, `declarations`, `artifacts` y `config_snapshot`.
    """
    if not MANIFEST_PATH.exists():
        return {"version": None, "declarations": [], "artifacts": {}, "config_snapshot": None}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def save_manifest(manifest: dict) -> None:
    """Escribe el manifiesto.

    Parámetros:
        manifest (dict): manifiesto completo.

    Retorno:
        None
    """
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _cfg_value(cfg: dict, key: str) -> Any:
    """Lee el valor de una clave con punto desde la config."""
    from src.config import get_key

    return get_key(cfg, key)


def record_declarations(keys: list[str], cfg: dict) -> None:
    """Registra en el manifiesto el valor actual de cada clave con su marca de tiempo UTC.

    Si la clave ya está declarada en la misma versión con el mismo valor, se conserva la
    declaración original; con otro valor, el proceso se detiene (cambiar una declaración exige
    nueva versión, Principio VI).

    Parámetros:
        keys (list[str]): claves con puntos a declarar.
        cfg (dict): configuración actual.

    Retorno:
        None
    """
    manifest = load_manifest()
    version = cfg["version"]
    now = _iso(utc_now())
    for key in keys:
        value = _cfg_value(cfg, key)
        prior = [
            d for d in manifest["declarations"] if d["key"] == key and d.get("version") == version
        ]
        if prior:
            if prior[-1]["value"] != value:
                raise RuntimeError(
                    f"La clave '{key}' ya está declarada en la versión {version} con otro valor; "
                    "cambiarla exige una nueva versión."
                )
            continue
        manifest["declarations"].append(
            {"key": key, "value": value, "declared_at": now, "version": version}
        )
    manifest["version"] = version
    save_manifest(manifest)


def _declaration(manifest: dict, key: str, version: str) -> dict | None:
    """Devuelve la última declaración de `key` en la versión dada, o None."""
    hits = [d for d in manifest["declarations"] if d["key"] == key and d.get("version") == version]
    return hits[-1] if hits else None


def require_declarations(keys: list[str], stage_start: datetime, cfg: dict) -> None:
    """Exige que cada clave esté declarada antes del inicio de la etapa y con el valor actual.

    Parámetros:
        keys (list[str]): claves con puntos.
        stage_start (datetime): inicio de la etapa (UTC).
        cfg (dict): configuración actual.

    Retorno:
        None. Lanza RuntimeError si alguna condición falla.
    """
    manifest = load_manifest()
    for key in keys:
        decl = _declaration(manifest, key, cfg["version"])
        if decl is None:
            raise RuntimeError(f"La clave '{key}' no está declarada antes de la etapa.")
        if _parse(decl["declared_at"]) >= stage_start:
            raise RuntimeError(f"La clave '{key}' se declaró después del inicio de la etapa.")
        if decl["value"] != _cfg_value(cfg, key):
            raise RuntimeError(
                f"La clave '{key}' cambió desde su declaración; cambiarla exige una nueva versión."
            )


def require_declared_before(keys: list[str], artifact_name: str) -> None:
    """Exige que cada clave se haya declarado antes del congelamiento de un artefacto.

    Parámetros:
        keys (list[str]): claves con puntos.
        artifact_name (str): nombre del artefacto en el manifiesto (p. ej. `matrix_y5`).

    Retorno:
        None. Lanza RuntimeError si el artefacto no existe o alguna declaración es posterior.
    """
    manifest = load_manifest()
    entry = manifest["artifacts"].get(artifact_name)
    if entry is None:
        raise RuntimeError(f"El artefacto '{artifact_name}' no está congelado.")
    frozen_at = _parse(entry["frozen_at"])
    for key in keys:
        decl = _declaration(manifest, key, entry.get("version"))
        if decl is None or _parse(decl["declared_at"]) >= frozen_at:
            raise RuntimeError(
                f"La clave '{key}' no se declaró antes de congelar '{artifact_name}'."
            )


def _write(obj: Any, path: Path) -> None:
    """Escribe un DataFrame como Parquet o cualquier otro objeto como JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(path, index=False, engine="pyarrow")
    else:
        path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def freeze(obj: Any, name: str, path: str | Path, cfg: dict) -> str:
    """Escribe un artefacto una sola vez y lo registra en el manifiesto (Principio V).

    Parámetros:
        obj (Any): DataFrame (se guarda en Parquet) o dict/list (se guarda en JSON).
        name (str): nombre del artefacto en el manifiesto.
        path (str | Path): ruta de destino.
        cfg (dict): configuración usada para producirlo.

    Retorno:
        str: SHA-256 del archivo escrito.
    """
    manifest = load_manifest()
    version = cfg["version"]
    existing = manifest["artifacts"].get(name)
    if existing is not None and existing.get("version") == version:
        raise RuntimeError(
            f"El artefacto '{name}' ya está congelado en la versión {version}; "
            "no se recalcula (Principio V)."
        )
    path = Path(path)
    _write(obj, path)
    digest = sha256_file(path)
    manifest["artifacts"][name] = {
        "path": path.as_posix(),
        "sha256": digest,
        "config_sha256": sha256_obj(cfg),
        "frozen_at": _iso(utc_now()),
        "version": version,
    }
    manifest["version"] = version
    manifest["config_snapshot"] = cfg
    save_manifest(manifest)
    return digest


def load_frozen(name: str) -> Any:
    """Carga un artefacto congelado tras verificar su hash.

    Parámetros:
        name (str): nombre del artefacto en el manifiesto.

    Retorno:
        Any: DataFrame (Parquet) o dict/list (JSON).
    """
    entry = load_manifest()["artifacts"].get(name)
    if entry is None:
        raise RuntimeError(f"El artefacto '{name}' no está congelado.")
    path = Path(entry["path"])
    if not path.exists() or sha256_file(path) != entry["sha256"]:
        raise RuntimeError(f"El hash de '{name}' no coincide con el manifiesto.")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest() -> None:
    """Verifica el hash de todos los artefactos congelados.

    Retorno:
        None. Lanza RuntimeError ante el primer hash que no coincida.
    """
    for name, entry in load_manifest()["artifacts"].items():
        path = Path(entry["path"])
        if not path.exists() or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"El hash de '{name}' no coincide con el manifiesto.")


class TestLedger:
    """Registro de evaluaciones en test: cada `model_id` se evalúa una sola vez (SC-007).

    Justificación (Principio III, condición c): mantiene estado que persiste entre ejecuciones.
    """

    __test__ = False  # Evita que pytest lo confunda con una clase de pruebas.

    def __init__(self, path: str | Path = LEDGER_PATH) -> None:
        """Abre el ledger.

        Parámetros:
            path (str | Path): ruta de `test_ledger.json`.

        Retorno:
            None
        """
        self.path = Path(path)

    def entries(self) -> list[dict]:
        """Devuelve las entradas registradas.

        Retorno:
            list[dict]: entradas con `model_id`, `evaluated_at` y hashes.
        """
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def has(self, model_id: str) -> bool:
        """Indica si el modelo ya se evaluó.

        Parámetros:
            model_id (str): identificador del modelo.

        Retorno:
            bool: True si ya tiene entrada.
        """
        return any(e["model_id"] == model_id for e in self.entries())

    def record(self, model_id: str, frozen_models_sha256: str, targets_sha256: str) -> None:
        """Registra una evaluación; se detiene si el modelo ya estaba registrado.

        Parámetros:
            model_id (str): identificador del modelo.
            frozen_models_sha256 (str): hash del archivo de modelos congelados.
            targets_sha256 (str): hash de la lista objetivo.

        Retorno:
            None
        """
        entries = self.entries()
        if any(e["model_id"] == model_id for e in entries):
            raise RuntimeError(f"El modelo '{model_id}' ya se evaluó en test (SC-007).")
        entries.append(
            {
                "model_id": model_id,
                "evaluated_at": _iso(utc_now()),
                "frozen_models_sha256": frozen_models_sha256,
                "targets_sha256": targets_sha256,
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
