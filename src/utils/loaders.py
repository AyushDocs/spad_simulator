from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import numpy as np
import pint

from ._exceptions import ConfigError
from ..core.units import Q, ureg

# Default data directory (set via set_data_dir or DATA_DIR env var)
_DATA_DIR: str | None = None


# ----------------------------------------------------------------
# typed data containers
# ----------------------------------------------------------------

def _to_pp(value: float | int | pint.Quantity, default_unit: str = "1") -> pint.Quantity:
    """Coerce *value* to pint Quantity if not already one."""
    if isinstance(value, pint.Quantity):
        return value
    return Q(value, default_unit)


def _to_pp_dict(d: dict[str, Any], default_unit: str = "1") -> dict[str, pint.Quantity]:
    """Coerce dict values to pint Quantity."""
    return {k: _to_pp(v, default_unit) for k, v in d.items()}


@dataclass
class MaterialData:
    name: str
    eps_r: pint.Quantity
    Eg_0K: pint.Quantity
    varshni_alpha: pint.Quantity
    varshni_beta: pint.Quantity
    Nc_300K: pint.Quantity
    Nv_300K: pint.Quantity
    dos_gamma: pint.Quantity
    mu_n: pint.Quantity
    mu_p: pint.Quantity
    vsat_n: pint.Quantity
    vsat_p: pint.Quantity
    mc: pint.Quantity
    mh: pint.Quantity
    ionization_e: dict[str, pint.Quantity]
    ionization_h: dict[str, pint.Quantity]
    tau_n: pint.Quantity
    tau_p: pint.Quantity

    def __post_init__(self) -> None:
        for field_name in (
            "eps_r", "Eg_0K", "varshni_alpha", "varshni_beta",
            "Nc_300K", "Nv_300K", "dos_gamma",
            "mu_n", "mu_p", "vsat_n", "vsat_p",
            "mc", "mh", "tau_n", "tau_p",
        ):
            val = getattr(self, field_name)
            if not isinstance(val, pint.Quantity):
                setattr(self, field_name, _to_pp(val))
        self.ionization_e = _to_pp_dict(self.ionization_e)
        self.ionization_h = _to_pp_dict(self.ionization_h)


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


def _get_pp(root: ET.Element, tag: str) -> pint.Quantity:
    """Parse a float value wrapped in pint Quantity using its ``unit`` attribute."""
    el = _get(root, tag)
    val = _float(el.text)
    unit = el.attrib.get("unit", "1")
    return Q(val, unit)


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
        ion_map: dict[str, dict[str, pint.Quantity]] = {}
        for ion in ion_els:
            carrier = ion.attrib.get("carrier", "")
            ion_map[carrier] = {
                "lambda0": _get_pp(ion, "lambda0"),
                "ER0": _get_pp(ion, "ER0"),
                "hw_meV": _get_pp(ion, "hw_meV"),
                "Eth": _get_pp(ion, "Eth"),
            }

        materials[name] = MaterialData(
            name=name,
            eps_r=_get_pp(mat_el, "property/[@name='eps_r']"),
            Eg_0K=_get_pp(mat_el, "property/[@name='Eg_0K']"),
            varshni_alpha=_get_pp(mat_el, "varshni/alpha"),
            varshni_beta=_get_pp(mat_el, "varshni/beta"),
            Nc_300K=_get_pp(mat_el, "dos/Nc_300K"),
            Nv_300K=_get_pp(mat_el, "dos/Nv_300K"),
            dos_gamma=_get_pp(mat_el, "dos/gamma"),
            mu_n=_get_pp(mat_el, "mobility/mu_n"),
            mu_p=_get_pp(mat_el, "mobility/mu_p"),
            vsat_n=_get_pp(mat_el, "saturation_velocity/vsat_n"),
            vsat_p=_get_pp(mat_el, "saturation_velocity/vsat_p"),
            mc=_get_pp(mat_el, "effective_mass/mc"),
            mh=_get_pp(mat_el, "effective_mass/mh"),
            ionization_e=ion_map.get("electron", {}),
            ionization_h=ion_map.get("hole", {}),
            tau_n=_get_pp(mat_el, "lifetime/tau_n"),
            tau_p=_get_pp(mat_el, "lifetime/tau_p"),
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
