import flet as ft
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import LinearNDInterpolator
import matplotlib
import matplotlib.pyplot as plt
import io
import base64

# Force Matplotlib to use non-GUI backend
matplotlib.use("Agg")

# -----------------------------------------------------------------------------
# Data Models & Utils
# -----------------------------------------------------------------------------


class SlagProperties:
    def __init__(self, csv_path="slag_data.csv"):
        self.model_visc = None
        self.model_liq = None
        try:
            df = pd.read_csv(csv_path)
            # Ensure columns exist
            required = ["Basicity", "Alumina_Pct", "Viscosity", "Liquid_Fraction"]
            if not all(col in df.columns for col in required):
                # If mock data or real data missing, just warn
                print(f"CSV missing columns: {required}")
                return

            points = df[["Basicity", "Alumina_Pct"]].values
            self.model_visc = LinearNDInterpolator(points, df["Viscosity"].values)
            self.model_liq = LinearNDInterpolator(points, df["Liquid_Fraction"].values)
        except Exception as e:
            print(f"Error loading slag data: {e}")

    def predict(self, b2, alumina):
        """
        Returns (viscosity, liquid_fraction).
        """
        if self.model_visc is None or self.model_liq is None:
            return 0.0, 0.0

        v = self.model_visc(b2, alumina)
        l = self.model_liq(b2, alumina)

        if np.isnan(v):
            v = 0.0
        if np.isnan(l):
            l = 0.0

        return float(v), float(l)


def generate_ternary_plot(cao, sio2, al2o3):
    """
    Generates a ternary plot (CaO-SiO2-Al2O3) with the point marked.
    Returns base64 string image.
    Vertices: CaO (Left), Al2O3 (Right), SiO2 (Top)
    """
    total = cao + sio2 + al2o3
    if total == 0:
        return None

    # Normalization
    f_cao = cao / total
    f_sio2 = sio2 / total
    f_al2o3 = al2o3 / total

    # Coordinates
    # CaO=(0,0), Al2O3=(1,0), SiO2=(0.5, sqrt(3)/2)
    x = f_al2o3 + 0.5 * f_sio2
    y = (np.sqrt(3) / 2) * f_sio2

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw Triangle
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, np.sqrt(3) / 2, 0]
    ax.plot(triangle_x, triangle_y, "k-", lw=1.5)

    # Labels
    ax.text(-0.05, -0.05, "CaO", fontsize=10, weight="bold")
    ax.text(1.05, -0.05, "Al2O3", fontsize=10, weight="bold")
    ax.text(0.5, np.sqrt(3) / 2 + 0.05, "SiO2", fontsize=10, weight="bold", ha="center")

    # Grid (Simplified)
    for i in range(1, 10):
        val = i / 10.0
        # Horizontal (constant SiO2)
        # y = val * sqrt(3)/2
        # ax.plot([val/2, 1-val/2], [val * np.sqrt(3)/2, val * np.sqrt(3)/2], 'k:', lw=0.5, alpha=0.3)
        pass  # Skip grid for clean look or add if needed

    # Plot Point
    ax.plot(x, y, "ro", markersize=10, markeredgecolor="black")
    ax.text(x + 0.03, y, "Final Slag", fontsize=9, color="red")

    # Save to Buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    return data


# -----------------------------------------------------------------------------
# Optimization Engine
# -----------------------------------------------------------------------------


def solve_optimization(
    carry_mass,
    carry_chem,
    materials,
    target_b2,
    target_mgo,
    target_max_caf2,
    deox_products,
    minimize_cost=False,
):
    """
    Args:
        deox_products: {'Al2O3': kg, 'SiO2': kg} generated from deoxidation.
        minimize_cost: Bool, if True minimize Price*Mass.
    """
    # 7 Oxides
    oxide_list = ["FeO", "CaO", "MgO", "SiO2", "Al2O3", "MnO", "CaF2"]

    # Effective mass of specific oxide in carry-over (including deoxidation products)
    def get_carry_oxide_mass(oxide):
        base_mass = carry_mass * (carry_chem.get(oxide, 0.0) / 100.0)
        added_mass = deox_products.get(oxide, 0.0)
        return base_mass + added_mass

    # Effective Total Carry Mass (Initial Slag + Deox Oxides)
    eff_carry_mass = carry_mass + sum(deox_products.values())

    n_materials = len(materials)
    x0 = np.ones(n_materials)
    bounds = [(0, None) for _ in range(n_materials)]

    # --- Constraints ---

    # 1. Basicity: CaO >= Target * SiO2
    def constraint_basicity(x):
        mass_cao = get_carry_oxide_mass("CaO")
        mass_sio2 = get_carry_oxide_mass("SiO2")

        for i, mass in enumerate(x):
            mat_chem = materials[i]["chem"]
            mass_cao += mass * (mat_chem.get("CaO", 0.0) / 100.0)
            mass_sio2 += mass * (mat_chem.get("SiO2", 0.0) / 100.0)

        if mass_sio2 == 0:
            return mass_cao
        return mass_cao - (target_b2 * mass_sio2)

    # 2. MgO Min
    def constraint_mgo(x):
        total_mass = eff_carry_mass + np.sum(x)
        mass_mgo = get_carry_oxide_mass("MgO")
        for i, mass in enumerate(x):
            mass_mgo += mass * (materials[i]["chem"].get("MgO", 0.0) / 100.0)
        return mass_mgo - ((target_mgo / 100.0) * total_mass)

    # 3. CaF2 Max: CaF2 <= Target%
    # Target% * Total - CaF2 >= 0
    def constraint_caf2(x):
        total_mass = eff_carry_mass + np.sum(x)
        mass_caf2 = get_carry_oxide_mass("CaF2")
        for i, mass in enumerate(x):
            mass_caf2 += mass * (materials[i]["chem"].get("CaF2", 0.0) / 100.0)
        return ((target_max_caf2 / 100.0) * total_mass) - mass_caf2

    cons = [
        {"type": "ineq", "fun": constraint_basicity},
        {"type": "ineq", "fun": constraint_mgo},
        {"type": "ineq", "fun": constraint_caf2},
    ]

    # Objective
    def objective(x):
        if minimize_cost:
            cost = 0.0
            for i, mass in enumerate(x):
                price = materials[i].get("price", 0.0)
                cost += mass * price
            return cost
        else:
            return np.sum(x)

    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=1e-4
    )

    # --- Results ---
    added_masses = {}
    final_chem = {}
    total_mass_final = eff_carry_mass + np.sum(result.x)

    for oxide in oxide_list:
        mass_oxide = get_carry_oxide_mass(oxide)
        for i, mass in enumerate(result.x):
            mass_oxide += mass * (materials[i]["chem"].get(oxide, 0.0) / 100.0)
        final_chem[oxide] = (
            (mass_oxide / total_mass_final) * 100.0 if total_mass_final > 0 else 0.0
        )

    cost_total = 0.0
    for i, mat in enumerate(materials):
        mass = max(0.0, result.x[i])
        added_masses[mat["name"]] = mass
        cost_total += mass * mat.get("price", 0.0)

    final_b2 = final_chem["CaO"] / final_chem["SiO2"] if final_chem["SiO2"] > 0 else 0.0

    return {
        "success": result.success,
        "message": result.message,
        "added_masses": added_masses,
        "final_mass": total_mass_final,
        "final_chem": final_chem,
        "final_b2": final_b2,
        "total_cost": cost_total,
    }


# -----------------------------------------------------------------------------
# GUI Components
# -----------------------------------------------------------------------------


class MaterialRow(ft.Container):
    def __init__(self, remove_callback, name="", defaults=None, price=0.1):
        super().__init__()
        self.remove_callback = remove_callback
        self.height = 70  # Increased for Price fields

        self.txt_name = ft.TextField(
            value=name, label="Name", width=100, height=40, text_size=12
        )
        self.txt_price = ft.TextField(
            value=str(price),
            label="$/kg",
            width=60,
            height=40,
            text_size=12,
            keyboard_type=ft.KeyboardType.NUMBER,
        )

        self.inputs = {}
        # 7 component list
        oxides = ["FeO", "CaO", "MgO", "SiO2", "Al2O3", "MnO", "CaF2"]
        chem_defaults = defaults if defaults else {}

        # Layout: Name | Price | Oxides... | Total | Delete
        row_controls = [self.txt_name, self.txt_price]

        # Total Indicator
        self.txt_total = ft.Text(
            value="sum: 0%",
            size=10,
            color=ft.Colors.GREEN_400,
            weight=ft.FontWeight.BOLD,
        )

        def on_change_chem(e):
            self.update_total()

        for oxide in oxides:
            val = str(chem_defaults.get(oxide, 0.0))
            tf = ft.TextField(
                value=val,
                label=oxide,
                width=60,
                height=40,
                text_size=11,
                content_padding=5,
                keyboard_type=ft.KeyboardType.NUMBER,
                on_change=on_change_chem,
            )
            self.inputs[oxide] = tf
            row_controls.append(tf)

        # Total Container
        row_controls.append(
            ft.Container(content=self.txt_total, width=50, alignment=ft.Alignment(0, 0))
        )

        # Delete Button (Red X)
        btn_del = ft.Container(
            content=ft.Text(
                "X", color=ft.Colors.RED_400, weight=ft.FontWeight.BOLD, size=18
            ),
            on_click=lambda e: self.remove_callback(self),
            width=30,
            height=30,
            alignment=ft.Alignment(0, 0),
        )
        row_controls.append(btn_del)

        self.content = ft.Row(
            controls=row_controls,
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=5,
        )
        self.update_total()

    def update_total(self):
        try:
            total = sum(
                float(v.value) if v.value else 0.0 for v in self.inputs.values()
            )
            self.txt_total.value = f"{total:.0f}%"
            if total > 100.1:
                self.txt_total.color = ft.Colors.RED_400
            else:
                self.txt_total.color = ft.Colors.GREEN_400
            if self.txt_total.page:
                self.txt_total.update()
        except:
            pass

    def get_data(self):
        try:
            chem = {}
            total = 0.0
            price = float(self.txt_price.value) if self.txt_price.value else 0.0
            for k, v in self.inputs.items():
                val = float(v.value) if v.value else 0.0
                if val < 0:
                    return None
                chem[k] = val
                total += val
            if total > 100.1:
                return None
            return {"name": self.txt_name.value, "chem": chem, "price": price}
        except:
            return None


def main(page: ft.Page):
    page.title = "Slag Optimizer Pro"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_width = 1200
    page.window_height = 900
    # page.scroll = ft.ScrollMode.AUTO  # DISABLED: Causes unbounded height error with TabBarView

    # -------------------------------------------------------------------------
    # UI Components creation
    # -------------------------------------------------------------------------

    # --- 1. Process Logic Setup ---
    # Deoxidation
    txt_do = ft.TextField(label="Dissolved Oxygen (ppm)", value="400", width=150)
    txt_al_add = ft.TextField(label="Al Added (kg)", value="0", width=150)
    txt_si_add = ft.TextField(label="FeSi Added (kg)", value="0", width=150)

    # Carry Over
    txt_carry_mass = ft.TextField(label="Slag Mass (kg)", value="1000", width=120)
    carry_inputs = {}
    oxides = ["FeO", "CaO", "MgO", "SiO2", "Al2O3", "MnO", "CaF2"]
    carry_defaults = {
        "FeO": 15,
        "CaO": 30,
        "MgO": 8,
        "SiO2": 30,
        "Al2O3": 10,
        "MnO": 7,
        "CaF2": 0,
    }
    carry_fields = []
    for ox in oxides:
        tf = ft.TextField(label=f"%{ox}", value=str(carry_defaults[ox]), width=70)
        carry_inputs[ox] = tf
        carry_fields.append(tf)

    # Targets
    txt_target_b2 = ft.TextField(label="Min B2 (CaO/SiO2)", value="2.0", width=130)
    txt_target_mgo = ft.TextField(label="Min %MgO", value="10.0", width=130)
    txt_target_caf2 = ft.TextField(label="Max %CaF2", value="5.0", width=130)

    # --- 2. Materials Setup ---
    material_rows_col = ft.Column(scroll=ft.ScrollMode.ALWAYS, expand=True)

    def add_material_row(e=None, name="", defaults=None, price=0.1):
        row = MaterialRow(remove_material_row, name, defaults, price)
        material_rows_col.controls.append(row)
        material_rows_col.update()

    def remove_material_row(row):
        material_rows_col.controls.remove(row)
        material_rows_col.update()

    btn_add_mat = ft.FilledButton("Add Material", icon="add", on_click=add_material_row)

    sw_minimize_cost = ft.Switch(label="Minimize Cost", value=False)

    # --- 3. Results Setup ---
    txt_status = ft.Text("Ready", color=ft.Colors.GREY)
    results_container = ft.Column()

    def run_opt(e):
        try:
            # Parse Deoxidation
            ppm_o = float(txt_do.value) if txt_do.value else 0.0
            kg_al = float(txt_al_add.value) if txt_al_add.value else 0.0
            kg_si = float(txt_si_add.value) if txt_si_add.value else 0.0

            # Generated Oxides
            # Al -> Al2O3 (ratio 102/54 = 1.89)
            # Si -> SiO2 (ratio 60/28 = 2.14)
            gen_al2o3 = kg_al * 1.89
            gen_sio2 = kg_si * 2.14

            # Parse Carry Over
            c_mass = float(txt_carry_mass.value)
            c_chem = {k: float(v.value) for k, v in carry_inputs.items()}

            # Parse Targets
            t_b2 = float(txt_target_b2.value)
            t_mgo = float(txt_target_mgo.value)
            t_caf2 = float(txt_target_caf2.value)

            # Parse Materials
            mats = []
            for ctrl in material_rows_col.controls:
                data = ctrl.get_data()
                if data:
                    mats.append(data)

            if not mats:
                page.snack_bar = ft.SnackBar(ft.Text("No materials!"))
                page.snack_bar.open = True
                page.update()
                return

            # Run Solver
            res = solve_optimization(
                c_mass,
                c_chem,
                mats,
                t_b2,
                t_mgo,
                t_caf2,
                {"Al2O3": gen_al2o3, "SiO2": gen_sio2},
                minimize_cost=sw_minimize_cost.value,
            )

            # Display
            if res["success"]:
                txt_status.value = "Optimization Successful"
                txt_status.color = ft.Colors.GREEN
            else:
                txt_status.value = f"Failed: {res['message']}"
                txt_status.color = ft.Colors.RED

            # Build Tables
            # Recipe
            recipe_rows = []
            for name, mass in res["added_masses"].items():
                if mass > 0.01:
                    recipe_rows.append(
                        ft.DataRow(
                            [
                                ft.DataCell(ft.Text(name)),
                                ft.DataCell(ft.Text(f"{mass:.2f}")),
                            ]
                        )
                    )

            # Chemistry
            chem_rows = []
            for ox, pct in res["final_chem"].items():
                chem_rows.append(
                    ft.DataRow(
                        [ft.DataCell(ft.Text(ox)), ft.DataCell(ft.Text(f"{pct:.2f}"))]
                    )
                )

            # Slag Prop Predictor
            try:
                sp = SlagProperties()
                visc, liq = sp.predict(
                    res["final_b2"], res["final_chem"].get("Al2O3", 0)
                )

                # Check NaNs
                if visc > 0:
                    # Visc Color
                    v_col = (
                        ft.Colors.GREEN
                        if visc < 3
                        else ft.Colors.ORANGE if visc < 5 else ft.Colors.RED
                    )
                    visc_txt = f"{visc:.2f} P"
                else:
                    v_col = ft.Colors.GREY
                    visc_txt = "N/A"

                prop_ui = ft.Row(
                    [
                        ft.Column(
                            [
                                ft.Text("Viscosity", size=10),
                                ft.Text(
                                    visc_txt,
                                    size=20,
                                    weight="bold",
                                    color=v_col,
                                ),
                            ]
                        ),
                        ft.Column(
                            [
                                ft.Text("Liquid %", size=10),
                                ft.ProgressBar(
                                    value=liq / 100.0,
                                    width=100,
                                    color=ft.Colors.BLUE,
                                ),
                                ft.Text(f"{liq:.1f}%"),
                            ]
                        ),
                    ],
                    spacing=20,
                )
            except:
                prop_ui = ft.Text("Data unavailable")

            # Phase Diagram
            img_base64 = generate_ternary_plot(
                res["final_chem"].get("CaO", 0),
                res["final_chem"].get("SiO2", 0),
                res["final_chem"].get("Al2O3", 0),
            )
            img_ctrl = (
                ft.Image(
                    src=f"data:image/png;base64,{img_base64}", width=300, height=250
                )
                if img_base64
                else ft.Container()
            )

            results_container.controls = [
                ft.Text(f"Final Mass: {res['final_mass']:.1f} kg"),
                ft.Text(f"Total Cost: ${res['total_cost']:.2f}"),
                ft.Divider(),
                ft.Row(
                    [
                        ft.Column(
                            [
                                ft.Text("Recipe"),
                                ft.DataTable(
                                    columns=[
                                        ft.DataColumn(ft.Text("Mat")),
                                        ft.DataColumn(ft.Text("Kg")),
                                    ],
                                    rows=recipe_rows,
                                ),
                            ]
                        ),
                        ft.Column(
                            [
                                ft.Text("Composition"),
                                ft.DataTable(
                                    columns=[
                                        ft.DataColumn(ft.Text("Ox")),
                                        ft.DataColumn(ft.Text("%")),
                                    ],
                                    rows=chem_rows,
                                ),
                            ]
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_AROUND,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
                ft.Divider(),
                ft.Text("Properties & Analysis", size=16, weight="bold"),
                ft.Row(
                    [prop_ui, img_ctrl], alignment=ft.MainAxisAlignment.SPACE_AROUND
                ),
            ]
            results_container.update()  # Explicit update for TabBarView content
            page.update()

        except Exception as ex:
            import traceback

            traceback.print_exc()  # Print full stack trace to terminal
            page.snack_bar = ft.SnackBar(ft.Text(f"Error: {str(ex)}"))
            page.snack_bar.open = True
            page.update()

    btn_calc = ft.FilledButton(
        "Run Optimization", icon="play_arrow", on_click=run_opt, height=50
    )

    # -------------------------------------------------------------------------
    # Manual Tabs Implementation (Robust - Controller Pattern)
    # -------------------------------------------------------------------------

    # Content Containers
    content_process = ft.Container(
        padding=20,
        content=ft.Column(
            [
                ft.Text("Deoxidation (Pre-Solver)", weight="bold"),
                ft.Row([txt_do, txt_al_add, txt_si_add]),
                ft.Divider(),
                ft.Text("Carry-Over Slag", weight="bold"),
                ft.Row([txt_carry_mass], wrap=True),
                ft.Row(carry_fields, wrap=True),
                ft.Divider(),
                ft.Text("Targets", weight="bold"),
                ft.Row([txt_target_b2, txt_target_mgo, txt_target_caf2]),
            ],
            scroll=ft.ScrollMode.AUTO,
        ),
    )

    content_materials = ft.Container(
        padding=20,
        content=ft.Column(
            [
                ft.Row([btn_add_mat, sw_minimize_cost]),
                ft.Divider(),
                ft.Container(
                    content=material_rows_col,
                    height=500,
                    border=ft.Border.all(1, ft.Colors.WHITE10),
                    border_radius=5,
                    padding=10,
                ),
            ]
        ),
    )

    content_results = ft.Container(
        padding=20,
        content=ft.Column(
            [
                ft.Row(
                    [btn_calc, txt_status], alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                ),
                ft.Divider(),
                results_container,
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,  # Ensure Column fills the Container
        ),
        expand=True,  # Ensure Container fills the TabView
    )

    # 1. TabBar (Labels)
    tab_bar = ft.TabBar(
        tabs=[
            ft.Tab(icon="settings", label="Process Inputs"),
            ft.Tab(icon="list", label="Materials & Cost"),
            ft.Tab(icon="analytics", label="Results"),
        ],
    )

    # 2. TabBarView (Content)
    tab_view = ft.TabBarView(
        controls=[
            content_process,
            content_materials,
            content_results,
        ]
    )

    # 3. Tabs Controller (High Level Wrapper)
    # Wraps Bar and View. `length` helps sync them.
    tabs_control = ft.Tabs(
        selected_index=0,
        length=3,
        content=ft.Column(
            [
                tab_bar,
                ft.Divider(height=1),
                ft.Container(content=tab_view, expand=True),
            ],
            expand=True,
        ),
        expand=True,
    )

    page.add(tabs_control)

    # Init defaults
    add_material_row(name="Lime", defaults={"CaO": 95}, price=0.12)
    add_material_row(name="Dolo", defaults={"CaO": 58, "MgO": 38}, price=0.15)
    add_material_row(name="Fluorspar", defaults={"CaF2": 90, "SiO2": 5}, price=0.45)


if __name__ == "__main__":
    ft.app(target=main)
