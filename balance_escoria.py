import flet as ft
import numpy as np
import pandas as pd
import math
from dataclasses import dataclass
from typing import Dict, Sequence, List, Optional
from scipy.optimize import minimize
from scipy.interpolate import LinearNDInterpolator
import matplotlib
matplotlib.use("Agg") # Backend no interactivo
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import io
import base64

# =============================================================================
# 1. MODELO DE VISCOSIDAD DE URBAIN (Tu código)
# =============================================================================

A_ALL = np.array([13.2, 30.5, -40.4, 60.8], dtype=float)
B_MG = np.array([15.9, -54.1, 138.0, -99.8], dtype=float)
B_CA = np.array([41.5, -117.2, 232.1, -156.4], dtype=float)
B_MN = np.array([20.0, 26.0, -110.3, 64.3], dtype=float)
C_MG = np.array([-18.6, 33.0, -112.0, 97.6], dtype=float)
C_CA = np.array([-45.0, 130.0, -298.6, 213.6], dtype=float)
C_MN = np.array([-25.6, -56.0, 186.2, -104.6], dtype=float)

MW: Dict[str, float] = {
    "SiO2": 60.0843, "Al2O3": 101.961, "CaO": 56.077, "MgO": 40.304,
    "FeO": 71.844, "Fe2O3": 159.688, "MnO": 70.937, "TiO2": 79.866,
    "Na2O": 61.979, "K2O": 94.196, "H2O": 18.01528, "P2O5": 141.943,
    "CaF2": 78.07 # Agregado para manejo de masa, aunque Urbain lo ignora
}

ORDER = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "Fe2O3", "MnO", "TiO2", "Na2O", "K2O", "H2O", "P2O5"]

@dataclass(frozen=True)
class UrbainResult:
    wt_norm: Dict[str, float]
    X: Dict[str, float]
    XG: float; XA: float; XM: float; X_ratio: float; alpha: float
    B_each: Dict[str, float]; B_mean: float
    T_C: np.ndarray; T_K: np.ndarray
    mu_Pa_s: np.ndarray

def normalize_wt_percent(wt: Dict[str, float]) -> Dict[str, float]:
    # Normaliza ignorando lo que no está en ORDER (ej. CaF2)
    s = sum(wt.get(ox, 0.0) for ox in ORDER)
    if s <= 0: return {ox: 0.0 for ox in ORDER} # Retorno seguro
    return {ox: 100.0 * wt.get(ox, 0.0) / s for ox in ORDER}

def wt_to_mole_fractions(wt_norm: Dict[str, float]) -> Dict[str, float]:
    n = np.array([wt_norm[ox] / MW[ox] for ox in ORDER], dtype=float)
    nt = float(n.sum())
    if nt <= 0: return {ox: 0.0 for ox in ORDER}
    Xvec = n / nt
    return {ox: float(Xvec[i]) for i, ox in enumerate(ORDER)}

def calc_B_from_coeff(a, b, c, alpha, XG) -> float:
    Bi = a + b * alpha + c * (alpha ** 2)
    B0, B1, B2, B3 = (float(Bi[0]), float(Bi[1]), float(Bi[2]), float(Bi[3]))
    return B0 + B1 * XG + B2 * (XG ** 2) + B3 * (XG ** 3)

def urbain_modified(wt_in: Dict[str, float], T_C: Sequence[float]) -> Optional[UrbainResult]:
    try:
        wt_norm = normalize_wt_percent(wt_in)
        X = wt_to_mole_fractions(wt_norm)
        
        XG = X["SiO2"] + X["P2O5"]
        XA = X["Al2O3"] + X["Fe2O3"]
        XM = sum(X[ox] for ox in ["CaO", "MgO", "MnO", "Na2O", "K2O", "FeO", "TiO2"])

        denom = (XA + XM)
        if denom <= 0: return None
        
        X_ratio = XG / (XG + XA + XM)
        alpha = XM / (XA + XM)

        B_each = {}
        if wt_in.get("MgO", 0.0) > 0: B_each["B_Mg"] = calc_B_from_coeff(A_ALL, B_MG, C_MG, alpha, XG)
        if wt_in.get("CaO", 0.0) > 0: B_each["B_Ca"] = calc_B_from_coeff(A_ALL, B_CA, C_CA, alpha, XG)
        if wt_in.get("MnO", 0.0) > 0: B_each["B_Mn"] = calc_B_from_coeff(A_ALL, B_MN, C_MN, alpha, XG)

        if not B_each: return None
        B_mean = float(np.mean(list(B_each.values())))

        T_C_arr = np.array(T_C, dtype=float).reshape(-1)
        T_K_arr = T_C_arr + 273.15
        
        A = math.exp(-(0.29 * B_mean + 11.57))
        mu_Pa_s = A * T_K_arr * np.exp(1000.0 * B_mean / T_K_arr)

        return UrbainResult(wt_norm, X, float(XG), float(XA), float(XM), float(X_ratio), float(alpha), B_each, B_mean, T_C_arr, T_K_arr, mu_Pa_s)
    except Exception as e:
        print(f"Urbain Error: {e}")
        return None

# =============================================================================
# 2. MODELO DE FASES (CSV INTERPOLATION)
# =============================================================================

class SlagPhaseModel:
    def __init__(self, csv_path):
        self.model_liq = None
        self.raw_data = None
        try:
            # Cargar CSV limpio: Alumina, Silica, Liquid Ratio
            df = pd.read_csv(csv_path)
            # Aseguramos nombres correctos
            df.columns = [c.strip() for c in df.columns]
            
            # Mapeo de columnas flexible
            col_al = next((c for c in df.columns if "Alumina" in c), None)
            col_si = next((c for c in df.columns if "Silica" in c), None)
            col_liq = next((c for c in df.columns if "Liquid" in c), None)

            if col_al and col_si and col_liq:
                self.raw_data = df[[col_al, col_si, col_liq]].copy()
                points = df[[col_al, col_si]].values
                values = df[col_liq].values
                self.model_liq = LinearNDInterpolator(points, values)
                print(" Modelo de Fases Cargado Exitosamente.")
            else:
                print(" Error: No se encontraron columnas de Alumina/Silica/Liquid en el CSV.")

        except Exception as e:
            print(f" Error cargando CSV de fases: {e}")

    def predict_liquid(self, alumina, silica):
        if self.model_liq is None: return 0.0
        val = self.model_liq(alumina, silica)
        if np.isnan(val): return 0.0
        return float(val) * 100.0 # Convertir ratio (0-1) a %

    def get_contour_data(self):
        return self.raw_data

# =============================================================================
# 3. INTERFAZ GRÁFICA (FLET)
# =============================================================================

def main(page: ft.Page):
    page.title = "Simulador de Escorias Siderúrgicas v2.0"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_width = 1300
    page.window_height = 900
    page.padding = 10

    # --- STATE ---
    phase_model = SlagPhaseModel("liquid_ratio_cleaned.csv")
    material_rows = []

    # --- TABS SETUP ---
    tabs = ft.Tabs(selected_index=0, animation_duration=300)
    
    # -------------------------------------------------------------------------
    # TAB 1: PROCESO (Desoxidación & Carry-Over)
    # -------------------------------------------------------------------------
    
    # Inputs Desoxidación
    txt_steel_mass = ft.TextField(label="Masa Acero (Ton)", value="100", width=120)
    txt_O_ppm = ft.TextField(label="Oxígeno Inicial (ppm)", value="450", width=120)
    txt_Al_add = ft.TextField(label="Al Añadido (kg)", value="30", width=100)
    txt_FeSi_add = ft.TextField(label="FeSi Añadido (kg)", value="50", width=100) # 75% Si
    txt_FeMn_add = ft.TextField(label="FeMn Añadido (kg)", value="100", width=100) # 80% Mn

    # Inputs Carry-Over
    txt_carry_mass = ft.TextField(label="Masa Escoria Olla (kg)", value="500", width=120)
    # Inputs Química Carry-Over
    oxides = ["FeO", "CaO", "MgO", "SiO2", "Al2O3", "MnO", "CaF2"]
    carry_defaults = {"FeO":15, "CaO":35, "MgO":8, "SiO2":25, "Al2O3":10, "MnO":7, "CaF2":0}
    inputs_carry = {ox: ft.TextField(label=f"%{ox}", value=str(carry_defaults.get(ox,0)), width=70, text_size=12) for ox in oxides}

    # Inputs Targets
    txt_target_b2 = ft.TextField(label="Target B2", value="2.0", width=100)
    txt_min_mgo = ft.TextField(label="Min %MgO", value="8.0", width=100)
    txt_max_caf2 = ft.TextField(label="Max %CaF2", value="5.0", width=100)
    txt_temp = ft.TextField(label="Temp (°C)", value="1600", width=100)

    tab_process = ft.Tab(
        text="1. Proceso & Inicio",
        content=ft.Column([
            ft.Container(
                content=ft.Column([
                    ft.Text("Desoxidación del Acero (Cálculo de Inclusiones)", size=16, weight="bold", color=ft.colors.BLUE_200),
                    ft.Row([txt_steel_mass, txt_O_ppm]),
                    ft.Row([txt_Al_add, txt_FeSi_add, txt_FeMn_add]),
                    ft.Text("Nota: Se asume Al 100%, FeSi 75%, FeMn 80%. El oxígeno se consume en orden: Al -> Si -> Mn", size=12, italic=True)
                ]), padding=10, border=ft.Border.all(1, ft.colors.WHITE24), border_radius=10
            ),
            ft.Container(
                content=ft.Column([
                    ft.Text("Escoria Carry-Over (Remanente del Horno)", size=16, weight="bold", color=ft.colors.ORANGE_200),
                    ft.Row([txt_carry_mass]),
                    ft.Row(list(inputs_carry.values()), wrap=True)
                ]), padding=10, border=ft.Border.all(1, ft.colors.WHITE24), border_radius=10
            ),
            ft.Container(
                content=ft.Column([
                    ft.Text("Objetivos de Optimización", size=16, weight="bold", color=ft.colors.GREEN_200),
                    ft.Row([txt_target_b2, txt_min_mgo, txt_max_caf2, txt_temp])
                ]), padding=10, border=ft.Border.all(1, ft.colors.WHITE24), border_radius=10
            )
        ], scroll=ft.ScrollMode.AUTO)
    )

    # -------------------------------------------------------------------------
    # TAB 2: MATERIALES (Carga)
    # -------------------------------------------------------------------------
    
    col_materials = ft.Column(scroll=ft.ScrollMode.ALWAYS, height=400)
    switch_cost = ft.Switch(label="Optimizar por Costo ($)", value=False)

    class MaterialRow(ft.Container):
        def __init__(self, remove_func, index, defaults=None):
            super().__init__()
            self.remove_func = remove_func
            self.inputs = {}
            
            self.txt_name = ft.TextField(value=f"Mat {index+1}", width=100, label="Nombre", height=40, text_size=12)
            self.txt_price = ft.TextField(value="0.1", width=60, label="$/kg", height=40, text_size=12)
            
            row_ctrls = [self.txt_name, self.txt_price]
            chem = defaults if defaults else {}
            
            for ox in oxides:
                val = str(chem.get(ox, 0.0))
                tf = ft.TextField(value=val, label=ox, width=60, height=40, text_size=12, content_padding=5)
                self.inputs[ox] = tf
                row_ctrls.append(tf)
            
            row_ctrls.append(ft.IconButton(icon=ft.icons.DELETE, icon_color="red", on_click=lambda _: self.remove_func(self)))
            
            self.content = ft.Row(row_ctrls, spacing=5, alignment=ft.MainAxisAlignment.START)

        def get_data(self):
            try:
                chem = np.array([float(self.inputs[ox].value) for ox in oxides])
                price = float(self.txt_price.value)
                return {"name": self.txt_name.value, "chem": chem, "price": price}
            except: return None

    def add_material(e=None, defaults=None):
        row = MaterialRow(lambda r: (material_rows.remove(r), col_materials.controls.remove(r), page.update()), len(material_rows), defaults)
        material_rows.append(row)
        col_materials.controls.append(row)
        page.update()

    # Botones Materiales
    btn_add = ft.FilledButton("Agregar Material", icon=ft.icons.ADD, on_click=lambda _: add_material())
    
    # Cargar defaults
    add_material(defaults={"CaO":90}) # Cal
    add_material(defaults={"CaO":55, "MgO":35}) # Dolomita
    add_material(defaults={"CaF2":85, "SiO2":10}) # Espato

    tab_materials = ft.Tab(
        text="2. Carga & Materiales",
        content=ft.Column([
            ft.Row([ft.Text("Lista de Aditivos", size=16, weight="bold"), switch_cost]),
            ft.Row([btn_add]),
            ft.Container(content=col_materials, border=ft.Border.all(1, ft.colors.WHITE10), border_radius=5, padding=5)
        ])
    )

    # -------------------------------------------------------------------------
    # TAB 3: RESULTADOS
    # -------------------------------------------------------------------------
    
    txt_status = ft.Text("Esperando cálculo...")
    results_container = ft.Column()
    img_plot = ft.Image(src_base64="", width=500, height=400, fit=ft.ImageFit.CONTAIN)

    def solve(e):
        txt_status.value = "Calculando..."
        page.update()
        
        try:
            # 1. LOGICA DE DESOXIDACIÓN
            # -------------------------
            steel_ton = float(txt_steel_mass.value)
            O_ppm_in = float(txt_O_ppm.value)
            
            # Kg de O total a remover (asumiendo todo reacciona para simplificar)
            kg_O_total = steel_ton * 1000 * (O_ppm_in / 1e6)
            
            kg_Al = float(txt_Al_add.value)
            kg_Si = float(txt_FeSi_add.value) * 0.75 # Grado FeSi
            kg_Mn = float(txt_FeMn_add.value) * 0.80 # Grado FeMn

            # Consumo de O por Al (2 Al + 3 O -> Al2O3) | Rel Mass: 54 Al consume 48 O (Ratio 0.888)
            # O mejor: 1 kg Al consume 0.89 kg O.
            O_consumed_Al = min(kg_O_total, kg_Al * (48/54))
            kg_Al2O3_gen = O_consumed_Al * (102/48)
            O_rem = kg_O_total - O_consumed_Al
            
            # Consumo por Si (Si + 2 O -> SiO2) | 28 Si consume 32 O (Ratio 1.14)
            O_consumed_Si = 0.0
            kg_SiO2_gen = 0.0
            if O_rem > 0:
                O_consumed_Si = min(O_rem, kg_Si * (32/28))
                kg_SiO2_gen = O_consumed_Si * (60/32)
                O_rem -= O_consumed_Si
            
            # Consumo por Mn (Mn + O -> MnO) | 55 Mn consume 16 O
            kg_MnO_gen = 0.0
            if O_rem > 0:
                O_consumed_Mn = min(O_rem, kg_Mn * (16/55))
                kg_MnO_gen = O_consumed_Mn * (71/16)

            # 2. PREPARAR SOLVER
            # ------------------
            # Masa base = Carry Over + Productos Desoxidación
            m_carry = float(txt_carry_mass.value)
            # Vector base: ["FeO", "CaO", "MgO", "SiO2", "Al2O3", "MnO", "CaF2"]
            base_chem = np.array([float(inputs_carry[ox].value) for ox in oxides])
            
            # Masa de cada oxido en carry over
            base_masses = (base_chem / 100.0) * m_carry
            
            # Sumar oxidos de desoxidación
            # SiO2 (idx 3), Al2O3 (idx 4), MnO (idx 5)
            base_masses[3] += kg_SiO2_gen
            base_masses[4] += kg_Al2O3_gen
            base_masses[5] += kg_MnO_gen
            
            total_base_mass = np.sum(base_masses)
            
            # Datos Materiales
            mats_data = [row.get_data() for row in material_rows if row.get_data()]
            if not mats_data: raise ValueError("No hay materiales")
            
            comps_matrix = np.array([m["chem"] for m in mats_data]).T # (7, N)
            prices = np.array([m["price"] for m in mats_data])
            
            # Targets
            t_b2 = float(txt_target_b2.value)
            t_mgo = float(txt_min_mgo.value)
            t_caf2_max = float(txt_max_caf2.value)

            # 3. EJECUTAR SOLVER (SCIPY)
            # --------------------------
            n_vars = len(mats_data)
            
            def mass_balance(x):
                added_mass_oxides = np.dot(comps_matrix, x)
                final_mass_oxides = base_masses + added_mass_oxides
                total_mass = np.sum(final_mass_oxides)
                return total_mass, final_mass_oxides

            def objective(x):
                if switch_cost.value: return np.sum(x * prices)
                return np.sum(x)

            # Constraints
            def cons_b2(x):
                _, oxs = mass_balance(x)
                # CaO(1) / SiO2(3)
                if oxs[3] < 0.1: return 0.0
                return (oxs[1]/oxs[3]) - t_b2
            
            def cons_mgo(x): # MgO(2)
                tot, oxs = mass_balance(x)
                return (oxs[2]/tot*100) - t_mgo

            def cons_caf2(x): # CaF2(6) <= Max
                tot, oxs = mass_balance(x)
                return t_caf2_max - (oxs[6]/tot*100)

            res = minimize(objective, np.full(n_vars, 10.0), bounds=[(0, None)]*n_vars, 
                           constraints=[{'type':'ineq', 'fun':cons_b2}, {'type':'ineq', 'fun':cons_mgo}, {'type':'ineq', 'fun':cons_caf2}], method='SLSQP')

            if not res.success:
                txt_status.value = f"Error en optimización: {res.message}"
                txt_status.color = "red"
                return

            # 4. POST-PROCESO
            # ---------------
            final_tot, final_oxs_kg = mass_balance(res.x)
            final_chem = (final_oxs_kg / final_tot) * 100
            final_dict = {ox: val for ox, val in zip(oxides, final_chem)}

            # Calcular Viscosidad Urbain
            temp_c = float(txt_temp.value)
            urbain_res = urbain_modified(final_dict, [temp_c])
            visc_val = urbain_res.mu_Pa_s[0] if urbain_res else 0.0
            
            # Calcular % Líquido (CSV)
            liq_pct = phase_model.predict_liquid(final_dict["Al2O3"], final_dict["SiO2"])

            # 5. GRAFICAR (TERNARIO + MAPA DE CALOR)
            # -------------------------------------
            try:
                plt.figure(figsize=(5, 4))
                
                # Coordenadas Ternarias (Normalizadas para plot)
                # Ejes: Izq=CaO, Der=SiO2, Arr=Al2O3
                # X = 0.5 * (2*SiO2 + Al2O3) / Total
                # Y = (sqrt(3)/2) * Al2O3 / Total
                
                # Datos de fondo (CSV)
                if phase_model.raw_data is not None:
                    # Normalizar datos CSV para ternario
                    df = phase_model.raw_data
                    # Asumimos que CaO es el resto aprox para pintar el triángulo
                    # Solo pintamos los puntos disponibles
                    sums = df.iloc[:,0] + df.iloc[:,1] + (100 - df.iloc[:,0] - df.iloc[:,1]) # Dummy sum
                    
                    # SiO2 (col 1), Al2O3 (col 0)
                    # Ojo: tu csv tiene Alumina col 0, Silica col 1
                    al_v = df.iloc[:,0].values
                    si_v = df.iloc[:,1].values
                    liq_v = df.iloc[:,2].values
                    
                    # Transformar a ternario
                    # Normalizamos localmente a 100 para la proyección
                    # (Asumiendo que el csv es balance CaO)
                    tot_v = 100.0 
                    x_v = 0.5 * (2 * si_v + al_v) / tot_v
                    y_v = (np.sqrt(3)/2) * al_v / tot_v
                    
                    # Mapa de Calor
                    plt.tricontourf(x_v, y_v, liq_v, levels=20, cmap='inferno')
                    plt.colorbar(label="Ratio Líquido (0-1)")

                # Triángulo Marco
                plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', lw=2)
                plt.text(-0.05, -0.05, 'CaO', weight='bold')
                plt.text(1.02, -0.05, 'SiO2', weight='bold')
                plt.text(0.48, 0.9, 'Al2O3', weight='bold')
                
                # Punto Final Escoria
                # Normalizar solo ternario para ploteo
                tern_sum = final_dict["CaO"] + final_dict["SiO2"] + final_dict["Al2O3"]
                if tern_sum > 0:
                    f_al = final_dict["Al2O3"] * (100/tern_sum)
                    f_si = final_dict["SiO2"] * (100/tern_sum)
                    px = 0.5 * (2 * f_si + f_al) / 100
                    py = (np.sqrt(3)/2) * f_al / 100
                    plt.plot(px, py, 'o', color='lime', markeredgecolor='black', markersize=10, label='Tu Escoria')
                    plt.legend(loc='upper right')

                plt.axis('off')
                plt.title(f"Proyección Ternaria (Fondo: Datos Thermo-Calc)")
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
                plt.close()
                buf.seek(0)
                img_plot.src_base64 = base64.b64encode(buf.read()).decode('utf-8')
                img_plot.update()

            except Exception as e:
                print(f"Plot Error: {e}")

            # 6. MOSTRAR RESULTADOS TEXTO
            # ---------------------------
            rows_recipe = []
            for i, val in enumerate(res.x):
                if val > 0.1:
                    rows_recipe.append(ft.DataRow([ft.DataCell(ft.Text(mats_data[i]["name"])), ft.DataCell(ft.Text(f"{val:.1f}"))]))

            rows_chem = [ft.DataRow([ft.DataCell(ft.Text(ox)), ft.DataCell(ft.Text(f"{final_dict[ox]:.2f}"))]) for ox in oxides]
            
            txt_status.value = "Cálculo Exitoso"
            txt_status.color = "green"
            
            # KPI Cards
            kpi_visc = ft.Container(content=ft.Column([
                ft.Text("Viscosidad (Urbain)", size=12), 
                ft.Text(f"{visc_val:.2f} Pa·s", size=20, weight="bold", color="cyan")
            ]), padding=10, bgcolor=ft.colors.BLACK45, border_radius=5)
            
            kpi_liq = ft.Container(content=ft.Column([
                ft.Text("Fase Líquida (CSV)", size=12), 
                ft.Text(f"{liq_pct:.1f} %", size=20, weight="bold", color="orange" if liq_pct < 100 else "green")
            ]), padding=10, bgcolor=ft.colors.BLACK45, border_radius=5)
            
            results_container.controls = [
                ft.Row([kpi_visc, kpi_liq]),
                ft.Divider(),
                ft.Row([
                    ft.DataTable(columns=[ft.DataColumn(ft.Text("Material")), ft.DataColumn(ft.Text("Kg Add"))], rows=rows_recipe),
                    ft.DataTable(columns=[ft.DataColumn(ft.Text("Oxido")), ft.DataColumn(ft.Text("% Final"))], rows=rows_chem)
                ], alignment=ft.MainAxisAlignment.SPACE_EVENLY),
                ft.Text(f"Productos Desoxidación: Al2O3={kg_Al2O3_gen:.1f}kg, SiO2={kg_SiO2_gen:.1f}kg, MnO={kg_MnO_gen:.1f}kg", italic=True)
            ]
            page.update()

        except Exception as ex:
            txt_status.value = f"Error Crítico: {ex}"
            txt_status.color = "red"
            page.update()

    btn_calc = ft.FilledButton("CALCULAR OPTIMIZACIÓN", on_click=solve, height=50)

    # LAYOUT FINAL
    tab_results = ft.Tab(text="3. Resultados & Análisis", content=ft.Column([
        txt_status,
        btn_calc,
        ft.Divider(),
        ft.Row([results_container, img_plot], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.START)
    ], scroll=ft.ScrollMode.AUTO))

    tabs.tabs = [tab_process, tab_materials, tab_results]
    page.add(tabs)

if __name__ == "__main__":
    ft.app(target=main)