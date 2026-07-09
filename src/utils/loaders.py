from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._exceptions import ConfigError

# Default data directory (set via set_data_dir or DATA_DIR env var)
_DATA_DIR: str | None = None


# ----------------------------------------------------------------
# typed data containers
# ----------------------------------------------------------------

@dataclass
class MaterialData:
    name: str
    eps_r: float
    Eg_0K: float
    varshni_alpha: float
    varshni_beta: float
    Nc_300K: float
    Nv_300K: float
    dos_gamma: float
    mu_n: float
    mu_p: float
    vsat_n: float
    vsat_p: float
    mc: float
    mh: float
    ionization_e: dict[str, float]
    ionization_h: dict[str, float]
    tau_n: float
    tau_p: float


@dataclass
class AbsorptionData:
    material: str
    wavelengths: np.ndarray
    alphas: np.ndarray


@dataclass
class DeviceSpec:
    name: str
    description: str
    nx: int
    temperature: float
    layers: list[dict[str, Any]]


# ----------------------------------------------------------------
# XML parser utilities
# ----------------------------------------------------------------


def set_data_dir(path: str | None) -> None:
    global _DATA_DIR
    _DATA_DIR = path


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    if _DATA_DIR is not None:
        joined = os.path.join(_DATA_DIR, path)
        if os.path.exists(joined):
            return joined
    return path


def _float(text: str | None) -> float:
    if text is None:
        raise ConfigError("Missing float value")
    return float(text)


def _int(text: str | None) -> int:
    if text is None:
        raise ConfigError("Missing int value")
    return int(text)


def _get(root: ET.Element, tag: str) -> ET.Element:
    el = root.find(tag)
    if el is None:
        raise ConfigError(f"Missing <{tag}> in <{root.tag}>")
    return el


def _get_text(root: ET.Element, tag: str) -> str:
    el = _get(root, tag)
    if el.text is None:
        raise ConfigError(f"Empty <{tag}>")
    return el.text.strip()


def _get_float(root: ET.Element, tag: str) -> float:
    return _float(_get_text(root, tag))


def _get_int(root: ET.Element, tag: str) -> int:
    return _int(_get_text(root, tag))


# ----------------------------------------------------------------
# materials loader
# ----------------------------------------------------------------

def load_materials(path: str) -> dict[str, MaterialData]:
    tree = ET.parse(_resolve(path))
    root = tree.getroot()
    if root.tag != "materials":
        raise ConfigError(f"Expected <materials> root, got <{root.tag}>")

    materials: dict[str, MaterialData] = {}
    for mat_el in root.findall("material"):
        name = mat_el.attrib.get("name", "")
        if not name:
            raise ConfigError("Material element missing 'name' attribute")

        ion_els = mat_el.findall("ionization")
        ion_map: dict[str, dict[str, float]] = {}
        for ion in ion_els:
            carrier = ion.attrib.get("carrier", "")
            ion_map[carrier] = {
                "lambda0": _get_float(ion, "lambda0"),
                "ER0": _get_float(ion, "ER0"),
                "hw_meV": _get_float(ion, "hw_meV"),
                "Eth": _get_float(ion, "Eth"),
            }

        materials[name] = MaterialData(
            name=name,
            eps_r=_get_float(mat_el, "property/[@name='eps_r']"),
            Eg_0K=_get_float(mat_el, "property/[@name='Eg_0K']"),
            varshni_alpha=_get_float(mat_el, "varshni/alpha"),
            varshni_beta=_get_float(mat_el, "varshni/beta"),
            Nc_300K=_get_float(mat_el, "dos/Nc_300K"),
            Nv_300K=_get_float(mat_el, "dos/Nv_300K"),
            dos_gamma=_get_float(mat_el, "dos/gamma"),
            mu_n=_get_float(mat_el, "mobility/mu_n"),
            mu_p=_get_float(mat_el, "mobility/mu_p"),
            vsat_n=_get_float(mat_el, "saturation_velocity/vsat_n"),
            vsat_p=_get_float(mat_el, "saturation_velocity/vsat_p"),
            mc=_get_float(mat_el, "effective_mass/mc"),
            mh=_get_float(mat_el, "effective_mass/mh"),
            ionization_e=ion_map.get("electron", {}),
            ionization_h=ion_map.get("hole", {}),
            tau_n=_get_float(mat_el, "lifetime/tau_n"),
            tau_p=_get_float(mat_el, "lifetime/tau_p"),
        )

    return materials


def load_absorption(path: str) -> dict[str, AbsorptionData]:
    tree = ET.parse(_resolve(path))
    root = tree.getroot()
    if root.tag != "absorption":
        raise ConfigError(f"Expected <absorption> root, got <{root.tag}>")

    result: dict[str, AbsorptionData] = {}
    for mat_el in root.findall("material"):
        name = mat_el.attrib.get("name", "")
        if not name:
            raise ConfigError("Absorption material missing 'name' attribute")

        wl_text = _get_text(mat_el, "wavelengths")
        al_text = _get_text(mat_el, "alphas")
        wavelengths = np.fromstring(wl_text, sep=" ", dtype=np.float64) * 1e-9
        alphas = np.fromstring(al_text, sep=" ", dtype=np.float64)
        if len(wavelengths) != len(alphas):
            raise ConfigError(
                f"Absorption data for '{name}': "
                f"{len(wavelengths)} wavelengths vs {len(alphas)} alphas"
            )

        result[name] = AbsorptionData(
            material=name,
            wavelengths=wavelengths,
            alphas=alphas,
        )

    return result


def load_device(path: str) -> DeviceSpec:
    tree = ET.parse(_resolve(path))
    root = tree.getroot()
    if root.tag != "device":
        raise ConfigError(f"Expected <device> root, got <{root.tag}>")

    meta_el = root.find("meta")
    description = ""
    if meta_el is not None:
        desc_el = meta_el.find("description")
        if desc_el is not None and desc_el.text:
            description = desc_el.text.strip()

    layers: list[dict[str, Any]] = []
    layers_el = _get(root, "layers")
    for layer_el in layers_el.findall("layer"):
        layers.append({
            "thickness_cm": _get_float(layer_el, "thickness"),
            "doping_type": _get_text(layer_el, "doping_type"),
            "doping_A": _get_float(layer_el, "doping_A"),
            "material": _get_text(layer_el, "material"),
        })

    return DeviceSpec(
        name=_get_text(root, "name"),
        description=description,
        nx=_get_int(root, "nx"),
        temperature=_get_float(root, "temperature"),
        layers=layers,
    )
