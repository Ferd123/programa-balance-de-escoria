import flet as ft
import numpy as np
import pandas as pd
import math
from dataclasses import dataclass
from typing import Dict, Sequence, List, Optional
from scipy.optimize import minimize
from scipy.interpolate import LinearNDInterpolator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

# =============================================================================
# 1. MODELO DE VISCOSIDAD DE URBAIN
# =============================================================================

A_ALL = np.array([13.2, 30.5, -40.4, 60.8], dtype=float)
B_MG = np.array([15.9, -54.1, 138.0, -99.8], dtype=float)
B_CA = np.array([41.5, -117.2, 232.1, -156.4], dtype=float)
B_MN = np.array([20.0, 26.0, -110.3, 64.3], dtype=float)
C_MG = np.array([-18.6, 33.0, -112.0, 97.6], dtype=float)
C_CA = np.array([-45.0, 130.0, -298.6, 213.6], dtype=float)
C_MN = np.array([-25.6, -56.0, 186.2, -104.6], dtype=float)

MW: Dict[str, float] = {
    "SiO2": 60.0843,
    "Al2O3": 101.961,
    "CaO": 56.077,
    "MgO": 40.304,
    "FeO": 71.844,
    "Fe2O3": 159.688,
    "MnO": 70.937,
    "TiO2": 79.866,
    "Na2O": 61.979,
    "K2O": 94.196,
    "H2O": 18.01528,
    "P2O5": 141.943,
    "CaF2": 78.07,
    "O": 16.00,
    "Al": 53.96,
    "Si": 28.09,
    "Mn": 54.94,
    "S": 32.06,
    "P": 30.97,
}

ORDER = [
    "SiO2",
    "Al2O3",
    "CaO",
    "MgO",
    "FeO",
    "Fe2O3",
    "MnO",
    "TiO2",
    "Na2O",
    "K2O",
    "H2O",
    "P2O5",
]


@dataclass(frozen=True)
class UrbainResult:
    wt_norm: Dict[str, float]
    X: Dict[str, float]
    XG: float
    XA: float
    XM: float
    X_ratio: float
    alpha: float
    B_each: Dict[str, float]
    B_mean: float
    T_C: np.ndarray
    T_K: np.ndarray
    mu_Pa_s: np.ndarray


def normalize_wt_percent(wt: Dict[str, float]) -> Dict[str, float]:
    s = sum(wt.get(ox, 0.0) for ox in ORDER)
    if s <= 0:
        return {ox: 0.0 for ox in ORDER}
    return {ox: 100.0 * wt.get(ox, 0.0) / s for ox in ORDER}


def wt_to_mole_fractions(wt_norm: Dict[str, float]) -> Dict[str, float]:
    n = np.array([wt_norm[ox] / MW[ox] for ox in ORDER], dtype=float)
    nt = float(n.sum())
    if nt <= 0:
        return {ox: 0.0 for ox in ORDER}
    Xvec = n / nt
    return {ox: float(Xvec[i]) for i, ox in enumerate(ORDER)}


def calc_B_from_coeff(a, b, c, alpha, XG) -> float:
    Bi = a + b * alpha + c * (alpha**2)
    B0, B1, B2, B3 = (float(Bi[0]), float(Bi[1]), float(Bi[2]), float(Bi[3]))
    return B0 + B1 * XG + B2 * (XG**2) + B3 * (XG**3)


def urbain_modified(
    wt_in: Dict[str, float], T_C: Sequence[float]
) -> Optional[UrbainResult]:
    try:
        wt_norm = normalize_wt_percent(wt_in)
        X = wt_to_mole_fractions(wt_norm)
        XG = X["SiO2"] + X["P2O5"]
        XA = X["Al2O3"] + X["Fe2O3"]
        XM = sum(X[ox] for ox in ["CaO", "MgO", "MnO", "Na2O", "K2O", "FeO", "TiO2"])
        denom = XA + XM
        if denom <= 0:
            return None
        X_ratio = XG / (XG + XA + XM)
        alpha = XM / (XA + XM)
        B_each = {}
        if wt_in.get("MgO", 0.0) > 0:
            B_each["B_Mg"] = calc_B_from_coeff(A_ALL, B_MG, C_MG, alpha, XG)
        if wt_in.get("CaO", 0.0) > 0:
            B_each["B_Ca"] = calc_B_from_coeff(A_ALL, B_CA, C_CA, alpha, XG)
        if wt_in.get("MnO", 0.0) > 0:
            B_each["B_Mn"] = calc_B_from_coeff(A_ALL, B_MN, C_MN, alpha, XG)
        if not B_each:
            return None
        B_mean = float(np.mean(list(B_each.values())))
        T_C_arr = np.array(T_C, dtype=float).reshape(-1)
        T_K_arr = T_C_arr + 273.15
        A = math.exp(-(0.29 * B_mean + 11.57))
        mu_Pa_s = A * T_K_arr * np.exp(1000.0 * B_mean / T_K_arr)
        return UrbainResult(
            wt_norm,
            X,
            float(XG),
            float(XA),
            float(XM),
            float(X_ratio),
            float(alpha),
            B_each,
            B_mean,
            T_C_arr,
            T_K_arr,
            mu_Pa_s,
        )
    except Exception:
        return None


# =============================================================================
# 2. MODELO DE FASES (CSV)
# =============================================================================


class SlagPhaseModel:
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        # Load Base Model (No CaF2)
        self.models["base"] = self._load_from_csv("liquid_ratio_cleaned.csv", "Base")
        # Load Fluidized Model (5% CaF2)
        self.models["caf2"] = self._load_from_csv(
            "liquid_ratio_cleaned_5CaF2.csv", "5% CaF2"
        )

    def _load_from_csv(self, path, name):
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip() for c in df.columns]
            col_al = next((c for c in df.columns if "Alumina" in c), None)
            col_si = next((c for c in df.columns if "Silica" in c), None)
            col_liq = next((c for c in df.columns if "Liquid" in c), None)

            if col_al and col_si and col_liq:
                points = df[[col_al, col_si]].values
                values = df[col_liq].values
                # Create interpolator
                model = LinearNDInterpolator(points, values)
                return {
                    "name": name,
                    "df": df[[col_al, col_si, col_liq]].copy(),
                    "model": model,
                    "valid": True,
                }
        except Exception as e:
            print(f"Error loading model {name} from {path}: {e}")

        return {"name": name, "df": None, "model": None, "valid": False}

    def get_model(self, strategy="economy"):
        # Returns: (model, dataframe, ternary_max, system_name)
        # Strategy: "economy" -> Base Model
        # Strategy: "fluidized" -> CaF2 Model

        if strategy == "fluidized" and self.models.get("caf2", {}).get("valid"):
            data = self.models["caf2"]
            return data["model"], data["df"], 85.0, "Sistema 5% CaF2"

        # Default to base (Economy)
        data = self.models.get("base", {})
        if data.get("valid"):
            return data["model"], data["df"], 90.0, "Sistema Base"

        return None, None, 90.0, "Sin Datos"

    def predict_liquid(self, alumina, silica, model):
        if model is None:
            return 0.0
        val = model(alumina, silica)
        if np.isnan(val):
            return 0.0
        return float(val) * 100.0


# =============================================================================
# 3. INTERFAZ GRÁFICA FINAL
# =============================================================================


def main(page: ft.Page):
    page.title = "Optimizador de Escorias v2.1"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_width = 1350
    page.window_height = 900
    page.padding = 10

    page.padding = 10

    phase_model = SlagPhaseModel()
    material_rows = []

    # --- PESTAÑA 1: PROCESO ---
    txt_steel_mass = ft.TextField(
        label="Masa Acero (Ton)",
        value="100",
        width=120,
        filled=False,
        border_color=ft.Colors.WHITE38,
    )
    txt_O_ppm = ft.TextField(
        label="O disuelto (ppm)",
        value="450",
        width=120,
        filled=False,
        border_color=ft.Colors.WHITE38,
    )
    txt_Al_add = ft.TextField(
        label="Al (kg)",
        value="30",
        width=100,
        filled=False,
        border_color=ft.Colors.WHITE38,
    )
    txt_FeSi_add = ft.TextField(
        label="FeSi (kg)",
        value="50",
        width=90,
        filled=False,
        border_color=ft.Colors.WHITE38,
    )
    txt_FeSi_grade = ft.TextField(
        label="%Si",
        value="75",
        width=60,
        filled=False,
        border_color=ft.Colors.WHITE38,
    )

    txt_FeMn_add = ft.TextField(
        label="FeMn (kg)",
        value="100",
        width=90,
        filled=False,
        border_color=ft.Colors.WHITE38,
    )
    txt_FeMn_grade = ft.TextField(
        label="%Mn",
        value="80",
        width=60,
        filled=False,
        border_color=ft.Colors.WHITE38,
    )

    # Parametros de Ajuste (Calibración)
    txt_gamma = ft.TextField(
        label="Gamma (Reduc. Carry)",
        value="0.5",
        width=120,
        tooltip="Fracción de FeO/MnO del carry-over que reacciona",
        filled=False,
        border_color=ft.Colors.WHITE38,
    )
    txt_fcap = ft.TextField(
        label="Efic. Captura",
        value="0.8",
        width=100,
        tooltip="Fracción de inclusiones atrapadas en escoria",
        filled=False,
        border_color=ft.Colors.WHITE38,
    )

    txt_carry_mass = ft.TextField(
        label="Masa Carry (kg)",
        value="500",
        width=140,
        filled=False,
        border_color=ft.Colors.WHITE38,
    )
    oxides = ["FeO", "CaO", "MgO", "SiO2", "Al2O3", "MnO", "CaF2"]
    carry_defaults = {
        "FeO": 15,
        "CaO": 35,
        "MgO": 8,
        "SiO2": 25,
        "Al2O3": 10,
        "MnO": 7,
        "CaF2": 0,
    }
    inputs_carry = {
        ox: ft.TextField(
            label=f"%{ox}",
            value=str(carry_defaults.get(ox, 0)),
            width=70,
            text_size=12,
            filled=False,
            border_color=ft.Colors.WHITE38,
        )
        for ox in oxides
    }

    txt_target_b2 = ft.TextField(
        label="Min B2",
        value="2.0",
        width=100,
        filled=False,
        border_color=ft.Colors.WHITE38,
    )
    txt_min_mgo = ft.TextField(
        label="Min %MgO",
        value="8.0",
        width=100,
        filled=False,
        border_color=ft.Colors.WHITE38,
    )
    txt_max_caf2 = ft.TextField(
        label="Max %CaF2",
        value="5.0",
        width=100,
        filled=False,
        border_color=ft.Colors.WHITE38,
    )
    txt_temp = ft.TextField(
        label="Temp (°C)",
        value="1600",
        width=100,
        filled=False,
        border_color=ft.Colors.WHITE38,
    )
    txt_min_mass = ft.TextField(
        label="Min Masa (kg)",
        value="1500",
        width=100,
        filled=False,
        border_color=ft.Colors.WHITE38,
    )

    content_process = ft.Column(
        [
            ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            "1. Desoxidación & Inclusiones (Realista)",
                            weight="bold",
                            color="blue200",
                        ),
                        ft.Row([txt_steel_mass, txt_O_ppm, txt_gamma, txt_fcap]),
                        ft.Row(
                            [
                                txt_Al_add,
                                ft.VerticalDivider(),
                                txt_FeSi_add,
                                txt_FeSi_grade,
                                ft.VerticalDivider(),
                                txt_FeMn_add,
                                txt_FeMn_grade,
                            ]
                        ),
                        ft.Text(
                            "Nota: Considera O disuelto + O reducible del Carry-Over (Gamma). Prioridad: Al > Si > Mn",
                            size=11,
                            italic=True,
                            color="grey",
                        ),
                    ]
                ),
                padding=15,
                border=ft.Border.all(1, ft.Colors.WHITE12),
                border_radius=8,
            ),
            ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            "2. Escoria Carry-Over", weight="bold", color="orange200"
                        ),
                        ft.Row([txt_carry_mass]),
                        ft.Row(list(inputs_carry.values()), wrap=True),
                    ]
                ),
                padding=15,
                border=ft.Border.all(1, ft.Colors.WHITE12),
                border_radius=8,
            ),
            ft.Container(
                content=ft.Column(
                    [
                        ft.Text("3. Targets", weight="bold", color="green200"),
                        ft.Row(
                            [
                                txt_target_b2,
                                txt_min_mgo,
                                txt_max_caf2,
                                txt_temp,
                                txt_min_mass,
                            ],
                            wrap=True,
                        ),
                    ]
                ),
                padding=15,
                border=ft.Border.all(1, ft.Colors.WHITE12),
                border_radius=8,
            ),
        ],
        scroll=ft.ScrollMode.AUTO,
    )

    # --- PESTAÑA 2: MATERIALES (CORREGIDA) ---
    col_materials = ft.Column(scroll=ft.ScrollMode.ALWAYS, height=450)
    switch_cost = ft.Switch(label="Minimizar Costo", value=False)

    # CLASE MATERIALROW CORREGIDA (ÚNICA DEFINICIÓN)
    class MaterialRow(ft.Container):
        def __init__(self, remove_func, index, defaults=None, price=0.1, name=None):
            super().__init__()
            self.remove_func = remove_func
            self.height = 60  # Altura fija para evitar el cuadro "larguísimo"

            style = {
                "height": 40,
                "text_size": 12,
                "content_padding": 10,
                "filled": False,
                "border_color": ft.Colors.WHITE38,
            }  # ESTILO TRANSPARENTE

            default_name = name if name else f"Mat {index+1}"
            self.txt_name = ft.TextField(
                value=default_name, width=100, label="Nombre", **style
            )
            self.txt_price = ft.TextField(
                value=str(price), label="$/kg", width=70, **style
            )

            self.inputs = {}
            chem = defaults if defaults else {}

            row_ctrls = [self.txt_name, self.txt_price]
            for ox in oxides:
                val = str(chem.get(ox, 0.0))
                tf = ft.TextField(
                    value=val,
                    label=ox,
                    width=65,
                    **style,
                    on_change=lambda e: self.update_total(),
                )
                self.inputs[ox] = tf
                row_ctrls.append(tf)

            self.txt_total = ft.Text("Total: 0%", size=12, color="white")
            row_ctrls.append(ft.Container(self.txt_total, padding=10))

            # Botón eliminar con ft.Icons (Validado)
            btn_del = ft.IconButton(
                icon=ft.Icons.DELETE_OUTLINE,
                icon_color="red400",
                tooltip="Eliminar Material",
                on_click=lambda _: self.remove_func(self),
            )
            row_ctrls.append(btn_del)

            self.content = ft.Row(
                row_ctrls,
                spacing=5,
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
            self.update_total(initial=True)  # Calc initial total

        def update_total(self, initial=False):
            try:
                total = sum([float(self.inputs[ox].value or 0) for ox in oxides])
                self.txt_total.value = f"Total: {total:.1f}%"
                self.txt_total.color = "red" if total > 100.1 else "green"
                if not initial:
                    self.txt_total.update()
            except:
                self.txt_total.value = "Error"
                if not initial:
                    self.txt_total.update()

        def get_data(self):
            try:
                chem = np.array([float(self.inputs[ox].value) for ox in oxides])
                return {
                    "name": self.txt_name.value,
                    "chem": chem,
                    "price": float(self.txt_price.value),
                }
            except:
                return None

    def add_material(e=None, defaults=None, price=0.1, name=None):
        def remove_material(row_instance):
            try:
                if row_instance in material_rows:
                    material_rows.remove(row_instance)
                if row_instance in col_materials.controls:
                    col_materials.controls.remove(row_instance)
                page.update()
            except Exception as e:
                print(f"Error removing row: {e}")

        row = MaterialRow(
            remove_material,
            len(material_rows),
            defaults,
            price,
            name,
        )
        material_rows.append(row)
        col_materials.controls.append(row)
        page.update()

    btn_add = ft.FilledButton(
        "Agregar Material", icon="add", on_click=lambda _: add_material()
    )
    add_material(defaults={"CaO": 92}, price=0.12, name="Cal Viva")
    add_material(defaults={"CaO": 55, "MgO": 35}, price=0.15, name="Dolomita")
    add_material(defaults={"CaF2": 85, "SiO2": 10}, price=0.25, name="Espato")

    content_materials = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Text("Lista de Materiales", size=16, weight="bold"),
                        switch_cost,
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ft.Container(
                    content=col_materials,
                    border=ft.Border.all(1, ft.Colors.WHITE10),
                    border_radius=5,
                    padding=5,
                    bgcolor=ft.Colors.BLACK12,
                ),
                ft.Row([btn_add], alignment=ft.MainAxisAlignment.END),
            ]
        )
    )

    # --- PESTAÑA 3: RESULTADOS ---
    txt_status = ft.Text("Esperando...")
    results_col = ft.Column()
    img_plot = ft.Image(src="", width=500, height=450, fit="contain")

    def solve(e):
        txt_status.value = "Calculando..."
        page.update()
        try:
            # 1. PARAMETROS DE ENTRADA
            M_steel = float(txt_steel_mass.value) * 1000
            ppm_O = float(txt_O_ppm.value)
            kg_O_dissolved = M_steel * (ppm_O / 1e6)

            gamma = float(txt_gamma.value)  # Factor reducción carry-over
            f_cap = float(txt_fcap.value)  # Factor captura inclusiones

            # Aleaciones disponibles
            kg_Al = float(txt_Al_add.value)
            kg_Si = float(txt_FeSi_add.value) * (float(txt_FeSi_grade.value) / 100.0)
            kg_Mn = float(txt_FeMn_add.value) * (float(txt_FeMn_grade.value) / 100.0)
            # 2. CÁLCULO OXÍGENO POTENCIAL DEL CARRY-OVER
            M_carry = float(txt_carry_mass.value)
            # Vector Carry Inicial [FeO, CaO, MgO, SiO2, Al2O3, MnO, CaF2]
            vec_carry_initial = (
                np.array([float(inputs_carry[ox].value) for ox in oxides])
                * M_carry
                / 100.0
            )

            # Oxígeno en FeO (MW O / MW FeO = 16/71.84 = 0.2227)
            O_from_FeO = vec_carry_initial[0] * (MW["O"] / MW["FeO"])
            # Oxígeno en MnO (MW O / MW MnO = 16/70.94 = 0.2255)
            O_from_MnO = vec_carry_initial[5] * (MW["O"] / MW["MnO"])

            O_reducible_carry = (O_from_FeO + O_from_MnO) * gamma

            # 3. LÓGICA DE DESOXIDACIÓN (Priority: Al > Si > Mn)
            O_total_demand = kg_O_dissolved + O_reducible_carry
            O_rem = O_total_demand
            # -- Paso A: Aluminio --
            # Capacidad Al: 2Al + 3O -> Al2O3. (54 Al comen 48 O). Ratio 0.888
            cap_O_Al = kg_Al * (48.0 / 53.96)
            O_consumed_Al = min(O_rem, cap_O_Al)
            gen_Al2O3 = O_consumed_Al * (101.96 / 48.0)
            O_rem -= O_consumed_Al

            # -- Paso B: Silicio --
            cap_O_Si = kg_Si * (32.0 / 28.09)
            O_consumed_Si = 0.0
            gen_SiO2 = 0.0
            if O_rem > 0:
                O_consumed_Si = min(O_rem, cap_O_Si)
                gen_SiO2 = O_consumed_Si * (60.08 / 32.0)
                O_rem -= O_consumed_Si

            # -- Paso C: Manganeso --
            cap_O_Mn = kg_Mn * (16.0 / 54.94)
            O_consumed_Mn = 0.0
            gen_MnO = 0.0
            if O_rem > 0:
                O_consumed_Mn = min(O_rem, cap_O_Mn)
                gen_MnO = O_consumed_Mn * (70.94 / 16.0)

            # 2. Balance Setup (Strict Immutability)
            m_carry = float(txt_carry_mass.value)

            # 4. BALANCE DE MASA DE LA REACCIÓN
            # ¿Cuánto O vino del Slag vs Acero?
            O_consumed_total = O_consumed_Al + O_consumed_Si + O_consumed_Mn
            O_taken_from_slag = max(0.0, O_consumed_total - kg_O_dissolved)

            # Repartimos la "pérdida" de O proporcionalmente entre FeO y MnO disponibles
            ratio_red = 0.0
            if O_reducible_carry > 0:
                ratio_red = O_taken_from_slag / O_reducible_carry

            mass_FeO_lost = (vec_carry_initial[0] * gamma) * ratio_red
            mass_MnO_lost = (vec_carry_initial[5] * gamma) * ratio_red

            # Vector de cambio por reacción (Deltas)
            # [FeO, CaO, MgO, SiO2, Al2O3, MnO, CaF2]
            vec_reaction_delta = np.zeros(7)
            vec_reaction_delta[0] = -mass_FeO_lost  # Se pierde FeO
            vec_reaction_delta[5] = -mass_MnO_lost  # Se pierde MnO

            # Sumamos lo generado (aplicando eficiencia de captura)
            vec_reaction_delta[3] += gen_SiO2 * f_cap  # SiO2
            vec_reaction_delta[4] += gen_Al2O3 * f_cap  # Al2O3
            vec_reaction_delta[5] += gen_MnO * f_cap  # MnO (Neto possible)

            # BASE PARA EL SOLVER (Carry + Reacción)
            # Use vec_carry_initial instead of creating vec_carry again
            vec_carry = vec_carry_initial
            vec_deox = (
                vec_reaction_delta  # Now contains both generation (+) and reduction (-)
            )

            # Base for Optimization
            base_optim = vec_carry + vec_deox
            # Ensure no negatives for solver stability
            base_optim = np.maximum(base_optim, 0.0)

            # OLD CODE BYPASS
            if False:
                vec_deox = np.zeros(7)

            # Base for Optimization (Sum of Carry + Deox)
            # This is the starting point for the solver, but we keep components separate for display.

            mats = [r.get_data() for r in material_rows if r.get_data()]
            if not mats:
                raise ValueError("Sin materiales")

            comps = np.array([m["chem"] for m in mats]).T / 100.0
            costs = np.array([m["price"] for m in mats])

            # Constraints Targets
            t_b2 = float(txt_target_b2.value)
            t_mgo = float(txt_min_mgo.value)
            t_caf2_limit = float(txt_max_caf2.value)
            t_mass = float(txt_min_mass.value)

            # --- OPTIMIZATION CORE ---
            # --- OPTIMIZATION CORE ---
            def run_optimization(min_caf2_constraint, max_caf2_constraint):
                # Local Objective Function
                def f_obj(x):
                    return np.sum(x * costs) if switch_cost.value else np.sum(x)

                # Local Constraints
                def constr(x):
                    # Current Total Mass = Base + Additions
                    vec_adds_temp = np.dot(comps, x)
                    fin = base_optim + vec_adds_temp
                    tot = np.sum(fin)
                    if tot == 0:
                        return [-1] * 5
                    return [
                        (fin[1] / fin[3]) - t_b2,  # B2 >= Target
                        (fin[2] / tot * 100) - t_mgo,  # MgO >= Target
                        max_caf2_constraint - (fin[6] / tot * 100),  # CaF2 <= Max Limit
                        (fin[6] / tot * 100) - min_caf2_constraint,  # CaF2 >= Min Limit
                        tot - t_mass,  # Mass >= Target
                    ]

                res_opt = minimize(
                    f_obj,
                    np.full(len(mats), 10.0),  # Initial guess
                    bounds=[(0, None)] * len(mats),
                    constraints=[
                        {"type": "ineq", "fun": lambda x: constr(x)[0]},
                        {"type": "ineq", "fun": lambda x: constr(x)[1]},
                        {"type": "ineq", "fun": lambda x: constr(x)[2]},
                        {"type": "ineq", "fun": lambda x: constr(x)[3]},
                        {"type": "ineq", "fun": lambda x: constr(x)[4]},
                    ],
                    method="SLSQP",
                )

                if not res_opt.success:
                    return None, None

                # Calculate Results if successful
                vec_adds_final = np.dot(comps, res_opt.x)
                final_kg_calc = base_optim + vec_adds_final
                final_pct_calc = final_kg_calc / np.sum(final_kg_calc) * 100
                d_fin_calc = {ox: v for ox, v in zip(oxides, final_pct_calc)}

                return res_opt, d_fin_calc

            # --- SMART SOLVER LOOP (2-Pass) ---

            # PASS 1: Economy Mode (Strict CaF2 limit)
            strategy_used = "Economy (Base)"
            active_model_name = "economy"
            limit_pass1 = min(0.5, t_caf2_limit)  # Enforce 0.5% max for economy

            res, d_fin = run_optimization(0.0, limit_pass1)

            # Check Quality using Base Model
            model_active, _, _, _ = phase_model.get_model("economy")
            liquid_pct = 0.0
            if res is not None:
                liquid_pct = phase_model.predict_liquid(
                    d_fin["Al2O3"], d_fin["SiO2"], model_active
                )

            # PASS 2: Fluidized Mode (If Pass 1 failed or slag too viscous)
            # Condition: (No Solution OR Liquid < 85%) AND User permits CaF2 > 0.5
            need_fluidizer = res is None or liquid_pct < 85.0
            can_add_fluorspar = t_caf2_limit > 0.5

            if need_fluidizer and can_add_fluorspar:
                # Retry with FORCE MIN 3% CaF2, Max 8%
                # This ensures the slag actually has CaF2 to match the diagram
                res2, d_fin2 = run_optimization(3.0, 8.0)

                if res2 is not None:
                    # Switch to Fluidized Strategy
                    strategy_used = "Fluidized (Min 3% CaF2)"
                    active_model_name = "fluidized"
                    model_active, _, _, _ = phase_model.get_model("fluidized")
                    liq2 = phase_model.predict_liquid(
                        d_fin2["Al2O3"], d_fin2["SiO2"], model_active
                    )

                    # Accept Pass 2 results
                    res = res2
                    d_fin = d_fin2
                    liquid_pct = liq2

            if res is None:
                raise ValueError("No se encontró solución factible (Infeasible)")

            final_kg = base_optim + np.dot(comps, res.x)  # Total Mass for KPI

            # Get model data for plotting based on final strategy
            model, df_ternary, ternary_max, sys_name = phase_model.get_model(
                active_model_name
            )
            liq = liquid_pct

            # Propiedades (Urbain)
            visc = 0.0
            urb = urbain_modified(d_fin, [float(txt_temp.value)])
            if urb:
                visc = urb.mu_Pa_s[0]

            # Graficar
            plt.figure(figsize=(5, 4.5))

            if df_ternary is not None:
                si, al, z = (
                    df_ternary.iloc[:, 1],
                    df_ternary.iloc[:, 0],
                    df_ternary.iloc[:, 2],
                )
                plt.tricontourf(
                    0.5 * (2 * si + al) / ternary_max,
                    (np.sqrt(3) / 2) * al / ternary_max,
                    z,
                    levels=15,
                    cmap="inferno",
                )
                plt.colorbar(label="Fracción Líquida")

            # Triángulo
            plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], "w-")
            plt.text(-0.05, -0.05, "CaO", color="white")
            plt.text(1.02, -0.05, "SiO2", color="white")
            plt.text(0.48, 0.9, "Al2O3", color="white")

            # Punto Calculado (Normalizado a ternary_max)
            ternary_sum = d_fin["CaO"] + d_fin["SiO2"] + d_fin["Al2O3"]
            if ternary_sum > 0:
                # Escalar componentes para que sumen ternary_max (90)
                # Esto asegura que el punto caiga dentro del triángulo si la suma real varía ligeramente
                ratio = ternary_max / ternary_sum
                f_si = d_fin["SiO2"] * ratio
                f_al = d_fin["Al2O3"] * ratio

                px = 0.5 * (2 * f_si + f_al) / ternary_max
                py = (np.sqrt(3) / 2) * f_al / ternary_max

                plt.plot(
                    px,
                    py,
                    "o",
                    color="lime",
                    markeredgecolor="black",
                    markersize=12,
                )

            plt.axis("off")
            plt.title(f"{sys_name} (Escala {ternary_max:.0f})", color="white")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", transparent=True)
            plt.close()
            buf.seek(0)
            img_plot.src = (
                f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
            )
            img_plot.update()

            # Tablas
            rows_mat = [
                ft.DataRow(
                    [
                        ft.DataCell(ft.Text(mats[i]["name"])),
                        ft.DataCell(ft.Text(f"{x:.1f}")),
                    ]
                )
                for i, x in enumerate(res.x)
                if x > 0.1
            ]
            rows_chem = [
                ft.DataRow(
                    [ft.DataCell(ft.Text(ox)), ft.DataCell(ft.Text(f"{d_fin[ox]:.1f}"))]
                )
                for ox in oxides
            ]

            results_col.controls = [
                ft.Row(
                    [
                        ft.Container(
                            ft.Column(
                                [
                                    ft.Text("Viscosidad", size=10),
                                    ft.Text(f"{visc:.3f} Pa·s", size=20, color="cyan"),
                                ]
                            ),
                            bgcolor="black45",
                            padding=10,
                            border_radius=5,
                        ),
                        ft.Container(
                            ft.Column(
                                [
                                    ft.Text("% Líquido", size=10),
                                    ft.Text(
                                        f"{liq:.1f} %",
                                        size=20,
                                        color="green" if liq > 90 else "orange",
                                    ),
                                ]
                            ),
                            bgcolor="black45",
                            padding=10,
                            border_radius=5,
                        ),
                        ft.Container(
                            ft.Column(
                                [
                                    ft.Text("Masa Total", size=10),
                                    ft.Text(
                                        f"{np.sum(final_kg):.0f} kg",
                                        size=20,
                                        color="white",
                                    ),
                                ]
                            ),
                            bgcolor="black45",
                            padding=10,
                            border_radius=5,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                ft.Row(
                    [
                        ft.DataTable(
                            columns=[
                                ft.DataColumn(ft.Text("Mat")),
                                ft.DataColumn(ft.Text("Kg")),
                            ],
                            rows=rows_mat,
                        ),
                        ft.DataTable(
                            columns=[
                                ft.DataColumn(ft.Text("Ox")),
                                ft.DataColumn(ft.Text("%")),
                            ],
                            rows=rows_chem,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                ),
                ft.Container(
                    ft.Text(
                        f"Estrategia: {strategy_used}",
                        size=12,
                        color="green" if "Economy" in strategy_used else "orange",
                    ),
                    alignment=ft.Alignment(0, 0),
                    padding=10,
                ),
            ]

            # --- UPDATE DETAILED BALANCE (STRICT SEPARATION) ---
            vec_adds = np.dot(comps, res.x)

            # Recalculate final_kg explicitly to match the columns
            final_kg_breakdown = vec_carry + vec_deox + vec_adds

            # Recalculate percentages based on this strict sum
            final_pct_breakdown = np.zeros_like(final_kg_breakdown)
            total_mass_check = np.sum(final_kg_breakdown)
            if total_mass_check > 0:
                final_pct_breakdown = final_kg_breakdown / total_mass_check * 100

            rows_det = []
            for i, ox in enumerate(oxides):
                rows_det.append(
                    ft.DataRow(
                        [
                            ft.DataCell(ft.Text(ox)),
                            ft.DataCell(
                                ft.Text(f"{vec_carry[i]:.1f}", color=ft.Colors.GREY_400)
                            ),
                            ft.DataCell(
                                ft.Text(
                                    f"{vec_deox[i]:.1f}",
                                    color=(
                                        ft.Colors.BLUE_200
                                        if vec_deox[i] >= 0
                                        else ft.Colors.RED_200
                                    ),
                                )
                            ),
                            ft.DataCell(
                                ft.Text(f"{vec_adds[i]:.1f}", color=ft.Colors.GREEN_200)
                            ),
                            ft.DataCell(
                                ft.Text(f"{final_kg_breakdown[i]:.1f}", weight="bold")
                            ),
                            ft.DataCell(
                                ft.Text(
                                    f"{final_pct_breakdown[i]:.1f}%",
                                    color=ft.Colors.CYAN_200,
                                )
                            ),
                        ]
                    )
                )
            dt_detailed.rows = rows_det

            tot_carry = np.sum(vec_carry)
            tot_deox = np.sum(vec_deox)
            tot_adds = np.sum(vec_adds)
            txt_summary_det.value = f"TOTALES -> Carry: {tot_carry:.0f} kg | Deox: {tot_deox:.0f} kg | Adiciones: {tot_adds:.0f} kg"

            txt_status.value = f"Éxito. Costo: ${sum(res.x*costs):.2f}"
            txt_status.color = "green"
            page.update()

        except Exception as ex:
            txt_status.value = f"Error: {ex}"
            txt_status.color = "red"
            page.update()

    btn_calc = ft.FilledButton("CALCULAR", icon="rocket_launch", on_click=solve)
    content_results = ft.Column(
        [
            ft.Row([btn_calc, txt_status], alignment="spaceBetween"),
            ft.Divider(),
            ft.Row(
                [results_col, img_plot], alignment="center", vertical_alignment="start"
            ),
        ],
        scroll=ft.ScrollMode.AUTO,
    )

    # --- PESTAÑA 4: DETALLE ---
    dt_detailed = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Oxido")),
            ft.DataColumn(ft.Text("Carry (kg)")),
            ft.DataColumn(ft.Text("Deox (kg)")),
            ft.DataColumn(ft.Text("Adds (kg)")),
            ft.DataColumn(ft.Text("TOTAL (kg)")),
            ft.DataColumn(ft.Text("% Final")),
        ],
        border=ft.Border.all(1, ft.Colors.WHITE10),
        vertical_lines=ft.Border.all(1, ft.Colors.WHITE10),
        heading_row_color=ft.Colors.BLACK45,
    )
    txt_summary_det = ft.Text("Esperando cálculo...", size=16, weight="bold")

    content_detailed = ft.Column(
        [
            ft.Text("Anatomía de la Escoria", size=20, weight="bold"),
            ft.Container(
                dt_detailed,
                border=ft.Border.all(1, ft.Colors.WHITE12),
                border_radius=10,
            ),
            ft.Divider(),
            ft.Container(
                txt_summary_det, bgcolor=ft.Colors.BLACK45, padding=10, border_radius=5
            ),
        ],
        scroll=ft.ScrollMode.AUTO,
    )

    # --- TABS CONTROL ---
    body = ft.Container(content=content_process, expand=True)

    def change_tab(idx):
        body.content = [
            content_process,
            content_materials,
            content_results,
            content_detailed,
        ][idx]
        for i, btn in enumerate(tabs_bar.controls):
            btn.style.bgcolor = ft.Colors.BLUE_900 if i == idx else None
            btn.style.color = ft.Colors.BLUE_200 if i == idx else ft.Colors.WHITE
            btn.update()
        body.update()

    tabs_bar = ft.Row(
        [
            ft.OutlinedButton(
                "1. Proceso",
                on_click=lambda _: change_tab(0),
                style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=0)),
            ),
            ft.OutlinedButton(
                "2. Carga",
                on_click=lambda _: change_tab(1),
                style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=0)),
            ),
            ft.OutlinedButton(
                "3. Resultados",
                on_click=lambda _: change_tab(2),
                style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=0)),
            ),
            ft.OutlinedButton(
                "4. Balance Detallado",
                on_click=lambda _: change_tab(3),
                style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=0)),
            ),
        ],
        spacing=0,
    )

    # Init Tab 1 Active Style
    tabs_bar.controls[0].style.bgcolor = ft.Colors.BLUE_900
    tabs_bar.controls[0].style.color = ft.Colors.BLUE_200

    page.add(ft.Column([tabs_bar, ft.Divider(height=1), body], expand=True))


if __name__ == "__main__":
    ft.app(target=main)
