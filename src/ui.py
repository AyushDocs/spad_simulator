"""Tkinter GUI for SPAD Simulator — select which studies to run via checkboxes + presets."""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from .utils._logging import set_log_level
from .utils.ingestion import DataIngestionConfig, DataIngestionService
from .studies import fields as flds
from .studies import dark_current as dc_st
from .studies import iv as iv_st
from .studies import pde as pde_st
from .studies import avalanche as av_st
from .studies import ionization as iz_st
from .studies import trap_density as td_st
from .studies import punch_breakdown as pb_st
from .studies import field_sweep as fs_st
from .studies import param_sweep as ps_st
from .studies import optimization as opt_st
from .studies import dcr_vs_pde as dcr_pde_st
from .studies import trigger_back_calculate as tbk_st

log = logging.getLogger("spad.ui")

# ---------------------------------------------------------------------------
# Study registry — each entry: (label, function, args_extra, depends_on_vbr)
# ---------------------------------------------------------------------------

StudyFn = Callable[..., Any]

STUDIES: list[dict] = [
    # --- Core (always runs) ---
    {"key": "device",      "group": "Core", "label": "Device structure",
     "fn": lambda sim, Vbr: flds.plot_device_structure(sim), "needs_vbr": False},
    {"key": "breakdown",   "group": "Core", "label": "Breakdown voltage",
     "fn": lambda sim, Vbr: flds.find_breakdown(sim), "needs_vbr": False},

    # --- I-V ---
    {"key": "field_sweep",        "group": "I-V", "label": "Field / potential profiles",
     "fn": lambda sim, Vbr: flds.run_field_sweep(sim, Vbr), "needs_vbr": True},
    {"key": "band_diagram",       "group": "I-V", "label": "Band diagram",
     "fn": lambda sim, Vbr: flds.run_band_diagram(sim, Vbr), "needs_vbr": True},
    {"key": "dark_current",       "group": "I-V", "label": "Dark current sweep",
     "fn": lambda sim, Vbr: dc_st.run_dark_current_sweep(sim, Vbr), "needs_vbr": True},
    {"key": "dc_components",      "group": "I-V", "label": "Dark current components vs Vex",
     "fn": lambda sim, Vbr: dc_st.run_dark_current_component_sweep(sim, Vbr), "needs_vbr": True},
    {"key": "iv_char",            "group": "I-V", "label": "I-V characteristic",
     "fn": lambda sim, Vbr: iv_st.run_iv_characteristic(sim, Vbr), "needs_vbr": True},
    {"key": "comprehensive_iv",   "group": "I-V", "label": "Comprehensive I-V",
     "fn": lambda sim, Vbr: iv_st.run_comprehensive_iv(sim, Vbr), "needs_vbr": True},
    {"key": "iv_sweep",           "group": "I-V", "label": "I-V sweep with decomposition",
     "fn": lambda sim, Vbr: ps_st.run_iv_sweep(sim, Vbr, decompose=True), "needs_vbr": True},

    # --- PDE ---
    {"key": "pde_spectrum",   "group": "PDE", "label": "PDE spectrum",
     "fn": lambda sim, Vbr: pde_st.run_pde_spectrum(sim, Vbr), "needs_vbr": True},
    {"key": "pde_vs_vex",     "group": "PDE", "label": "PDE vs excess voltage",
     "fn": lambda sim, Vbr: pde_st.run_pde_vs_vex(sim, Vbr), "needs_vbr": True},
    {"key": "absorption",     "group": "PDE", "label": "Absorption profile",
     "fn": lambda sim, Vbr: pde_st.run_absorption_profile(sim, Vbr), "needs_vbr": True},
    {"key": "pde_3d",         "group": "PDE", "label": "PDE 3D surface",
     "fn": lambda sim, Vbr: pde_st.run_pde_3d(sim, Vbr), "needs_vbr": True},

    # --- Trigger & Avalanche ---
    {"key": "trigger_profiles",   "group": "Trigger", "label": "Trigger probability profiles",
     "fn": lambda sim, Vbr: flds.run_trigger_profiles(sim, Vbr), "needs_vbr": True},
    {"key": "trigger_vs_vex",     "group": "Trigger", "label": "Trigger vs excess voltage",
     "fn": lambda sim, Vbr: flds.run_trigger_vs_vex(sim, Vbr), "needs_vbr": True},
    {"key": "afterpulsing",       "group": "Trigger", "label": "Afterpulsing",
     "fn": lambda sim, Vbr: av_st.run_afterpulsing(sim, Vbr), "needs_vbr": True},
    {"key": "excess_noise",       "group": "Trigger", "label": "Excess noise",
     "fn": lambda sim, Vbr: av_st.run_excess_noise(sim, Vbr), "needs_vbr": True},
    {"key": "jitter",             "group": "Trigger", "label": "Timing jitter",
     "fn": lambda sim, Vbr: av_st.run_jitter(sim, Vbr), "needs_vbr": True},
    {"key": "breakdown_prob",     "group": "Trigger", "label": "Breakdown probability vs Vex",
     "fn": lambda sim, Vbr: av_st.run_breakdown_prob_vs_vex(sim, Vbr), "needs_vbr": True},
    {"key": "dead_space",         "group": "Trigger", "label": "Dead space distribution",
     "fn": lambda sim, Vbr: av_st.run_dead_space_distribution(sim, Vbr), "needs_vbr": True},
    {"key": "avalanche_pulse",    "group": "Trigger", "label": "Avalanche current pulse",
     "fn": lambda sim, Vbr: av_st.run_avalanche_current_pulse(sim, Vbr), "needs_vbr": True},
    {"key": "quenching",          "group": "Trigger", "label": "Quenching waveform",
     "fn": lambda sim, Vbr: av_st.run_quenching_waveform(sim, Vbr), "needs_vbr": True},
    {"key": "trigger_back_calc",  "group": "Trigger", "label": "Trigger back-calc (validate Ptr)",
     "fn": lambda sim, Vbr: tbk_st.run_trigger_back_calculate(sim, Vbr), "needs_vbr": True},

    # --- Dark Current ---
    {"key": "trap_density",       "group": "Dark Current", "label": "Trap density sweep",
     "fn": lambda sim, Vbr: td_st.run_trap_density_iv(sim, Vbr), "needs_vbr": True},

    # --- Temperature ---
    {"key": "dcr_vs_temp",      "group": "Temperature", "label": "DCR vs temperature",
     "fn": lambda svc, Vbr: dc_st.run_dcr_vs_temp(svc, Vbr), "needs_vbr": True, "uses_svc": True},
    {"key": "pde_vs_temp",      "group": "Temperature", "label": "PDE vs temperature",
     "fn": lambda svc, Vbr: pde_st.run_pde_vs_temp(svc, Vbr), "needs_vbr": True, "uses_svc": True},
    {"key": "bv_vs_temp",       "group": "Temperature", "label": "Breakdown voltage vs temperature",
     "fn": lambda svc, Vbr: flds.run_breakdown_vs_temp(svc, Vbr), "needs_vbr": True, "uses_svc": True},
    {"key": "dc_comp_temp",     "group": "Temperature", "label": "Dark current components vs temp",
     "fn": lambda svc, Vbr: dc_st.run_dark_current_components_vs_temp(svc, Vbr), "needs_vbr": True, "uses_svc": True},

    # --- Ionization ---
    {"key": "ionization_field", "group": "Ionization", "label": "Ionization coeffs vs field",
     "fn": lambda sim, Vbr: iz_st.run_ionization_vs_field(sim, Vbr), "needs_vbr": True},
    {"key": "multiplication",   "group": "Ionization", "label": "Multiplication vs excess bias",
     "fn": lambda sim, Vbr: iz_st.run_multiplication_vs_vex(sim, Vbr), "needs_vbr": True},

    # --- Punch-through / Parametric ---
    {"key": "punch_breakdown",  "group": "Parametric", "label": "Punch-through / breakdown sweep",
     "fn": lambda sim, Vbr: pb_st.run_punch_breakdown_sweep(sim, Vbr), "needs_vbr": True},
    {"key": "sweep_abs_width",  "group": "Parametric", "label": "E-field vs absorption width",
     "fn": lambda sim, Vbr: fs_st.run_efield_vs_absorption_width(sim, Vbr), "needs_vbr": True},
    {"key": "sweep_mul_width",  "group": "Parametric", "label": "E-field vs multiplication width",
     "fn": lambda sim, Vbr: fs_st.run_efield_vs_multiplication_width(sim, Vbr), "needs_vbr": True},
    {"key": "sweep_abs_thick",  "group": "Parametric", "label": "Vbr vs absorption thickness",
     "fn": lambda sim, Vbr: ps_st.sweep_absorption_thickness(sim, Vbr), "needs_vbr": True},
    {"key": "sweep_mul_thick",  "group": "Parametric", "label": "Vbr vs multiplication thickness",
     "fn": lambda sim, Vbr: ps_st.sweep_multiplication_thickness(sim, Vbr), "needs_vbr": True},
    {"key": "sweep_charge",     "group": "Parametric", "label": "Vbr vs charge density",
     "fn": lambda sim, Vbr: ps_st.sweep_charge_density(sim, Vbr), "needs_vbr": True},
    {"key": "sweep_p_doping",   "group": "Parametric", "label": "Vbr vs p-layer doping",
     "fn": lambda sim, Vbr: ps_st.sweep_p_layer_doping(sim, Vbr), "needs_vbr": True},

    # --- Additional ---
    {"key": "peak_field",       "group": "Additional", "label": "Peak field vs bias",
     "fn": lambda sim, Vbr: flds.run_peak_field_vs_bias(sim, Vbr), "needs_vbr": True},
    {"key": "avalanche_map",    "group": "Additional", "label": "Avalanche probability map",
     "fn": lambda sim, Vbr: flds.run_avalanche_probability_map(sim, Vbr), "needs_vbr": True},

    # --- Optimization ---
    {"key": "optimize",         "group": "Optimization", "label": "PSO device optimization",
     "fn": lambda sim, Vbr: opt_st.run_optimize_device(sim, Vbr), "needs_vbr": True},

    # --- DCR vs PDE ---
    {"key": "dcr_vs_pde",      "group": "DCR vs PDE", "label": "DCR vs PDE (multiplication width sweep)",
     "fn": lambda sim, Vbr: dcr_pde_st.run_dcr_vs_pde(sim, Vbr), "needs_vbr": True},
]

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, list[str]] = {
    "Quick IV Only": [
        "device", "breakdown", "field_sweep", "dark_current", "iv_char",
    ],
    "Dark Current Analysis": [
        "device", "breakdown", "dark_current", "dc_components",
        "dcr_vs_temp", "dc_comp_temp", "trap_density",
    ],
    "PDE Analysis": [
        "device", "breakdown", "pde_spectrum", "pde_vs_vex",
        "absorption", "pde_3d", "pde_vs_temp",
    ],
    "Trigger & Avalanche": [
        "device", "breakdown", "trigger_profiles", "trigger_vs_vex",
        "afterpulsing", "excess_noise", "jitter",
        "breakdown_prob", "dead_space", "avalanche_pulse", "quenching",
    ],
    "Temperature Sweep": [
        "device", "breakdown", "dcr_vs_temp", "pde_vs_temp",
        "bv_vs_temp", "dc_comp_temp",
    ],
    "Ionization Study": [
        "device", "breakdown", "ionization_field", "multiplication",
    ],
    "Parametric Sweeps": [
        "device", "breakdown", "punch_breakdown",
        "sweep_abs_width", "sweep_mul_width",
        "sweep_abs_thick", "sweep_mul_thick",
        "sweep_charge", "sweep_p_doping",
    ],
    "Full Characterization": [s["key"] for s in STUDIES],
}


# ---------------------------------------------------------------------------
# UI Application
# ---------------------------------------------------------------------------

class SPADSimulatorUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("SPAD Simulator — Study Selector")
        self.root.geometry("1000x700")

        self._vars: dict[str, tk.BooleanVar] = {}
        self._checkbuttons: dict[str, ttk.Checkbutton] = {}
        self._running = False

        self._build_menu()
        self._build_main()

    # ---- Menu bar ----------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Quit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

    def _show_about(self) -> None:
        tk.messagebox.showinfo(
            "SPAD Simulator",
            "InGaAs/InP SAGCM SPAD Simulator (1D center-axis)\n\n"
            "Select studies via checkboxes and click Run.",
        )

    # ---- Main layout -------------------------------------------------------

    def _build_main(self) -> None:
        top_frame = ttk.Frame(self.root, padding=8)
        top_frame.pack(fill=tk.BOTH, expand=True)

        # -- Presets (top) --
        preset_frame = ttk.LabelFrame(top_frame, text="Presets", padding=6)
        preset_frame.pack(fill=tk.X, pady=(0, 8))

        for name in PRESETS:
            btn = ttk.Button(preset_frame, text=name,
                             command=lambda n=name: self._apply_preset(n))
            btn.pack(side=tk.LEFT, padx=3)

        # -- Left: checkboxes | Right: Run + Log --
        paned = ttk.PanedWindow(top_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(paned, padding=4)
        right = ttk.Frame(paned, padding=4)

        paned.add(left, weight=2)
        paned.add(right, weight=3)

        self._build_checkboxes(left)
        self._build_run_panel(right)

    # ---- Checkboxes --------------------------------------------------------

    def _build_checkboxes(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        groups: dict[str, list[dict]] = {}
        for s in STUDIES:
            groups.setdefault(s["group"], []).append(s)

        for group_name, items in groups.items():
            grp = ttk.LabelFrame(scroll_frame, text=group_name, padding=4)
            grp.pack(fill=tk.X, pady=2)

            all_var = tk.BooleanVar(value=False)
            row = ttk.Frame(grp)
            row.pack(fill=tk.X)
            ttk.Checkbutton(row, text=f"All {group_name}",
                            variable=all_var,
                            command=lambda g=group_name, av=all_var: self._toggle_group(g, av)).pack(side=tk.LEFT)

            for s in items:
                key = s["key"]
                var = tk.BooleanVar(value=(key == "device" or key == "breakdown"))
                self._vars[key] = var
                cb = ttk.Checkbutton(grp, text=s["label"], variable=var)
                cb.pack(fill=tk.X, padx=16, pady=1)
                self._checkbuttons[key] = cb

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _toggle_group(self, group: str, var: tk.BooleanVar) -> None:
        val = var.get()
        for s in STUDIES:
            if s["group"] == group:
                self._vars[s["key"]].set(val)

    # ---- Run panel ---------------------------------------------------------

    def _build_run_panel(self, parent: ttk.Frame) -> None:
        # -- Run button --
        self._run_btn = ttk.Button(parent, text="▶  Run Selected",
                                   command=self._run)
        self._run_btn.pack(pady=4)

        self._status_label = ttk.Label(parent, text="Ready", foreground="gray")
        self._status_label.pack(pady=2)

        # -- Log output --
        log_frame = ttk.LabelFrame(parent, text="Output Log", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        self._log_text = tk.Text(log_frame, wrap=tk.WORD, font=("Consolas", 9),
                                  state=tk.DISABLED, bg="#1e1e1e", fg="#d4d4d4")
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_scroll.set)

        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ---- Preset application ------------------------------------------------

    def _apply_preset(self, name: str) -> None:
        keys = PRESETS.get(name, [])
        for s in STUDIES:
            self._vars[s["key"]].set(s["key"] in keys)
        self._log(f"Preset applied: {name}")

    # ---- Run logic ---------------------------------------------------------

    def _run(self) -> None:
        if self._running:
            return
        selected = [s for s in STUDIES if self._vars[s["key"]].get()]
        if not selected:
            self._log("No studies selected.")
            return

        self._running = True
        self._run_btn.config(state=tk.DISABLED, text="⏳ Running...")
        self._status_label.config(text="Running...", foreground="blue")
        self._log(f"Starting simulation with {len(selected)} study(ies)...")

        cfg = DataIngestionConfig.from_defaults()
        svc = DataIngestionService(cfg)

        def task() -> None:
            set_log_level(logging.INFO)
            try:
                sim = svc.build_simulator()
            except Exception as e:
                self._log(f"Build failed: {e}")
                self._finish_run()
                return

            Vbr = None
            for s in selected:
                key = s["key"]
                self._log(f"→ {s['label']}...")
                try:
                    if key == "breakdown":
                        Vbr = flds.find_breakdown(sim)
                        self._log(f"  Vbr = {Vbr:.1f} V")
                    elif key == "device":
                        flds.plot_device_structure(sim)
                    else:
                        if Vbr is None:
                            self._log("  ⚠ Skipping — Vbr not yet computed")
                            continue
                        if s.get("uses_svc"):
                            s["fn"](svc, Vbr)
                        else:
                            s["fn"](sim, Vbr)
                    self._log("  ✓ done")
                except Exception as e:
                    self._log(f"  ✗ FAILED: {e}")

            self._log("Simulation complete.")
            self._finish_run()

        threading.Thread(target=task, daemon=True).start()

    def _finish_run(self) -> None:
        self.root.after(0, self._finish_run_ui)

    def _finish_run_ui(self) -> None:
        self._running = False
        self._run_btn.config(state=tk.NORMAL, text="▶  Run Selected")
        self._status_label.config(text="Done", foreground="green")

    # ---- Logging -----------------------------------------------------------

    def _log(self, msg: str) -> None:
        def append() -> None:
            self._log_text.config(state=tk.NORMAL)
            self._log_text.insert(tk.END, msg + "\n")
            self._log_text.see(tk.END)
            self._log_text.config(state=tk.DISABLED)
            self.root.update_idletasks()
        self.root.after(0, append)

    # ---- Entry point -------------------------------------------------------

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = SPADSimulatorUI()
    app.run()


if __name__ == "__main__":
    main()
