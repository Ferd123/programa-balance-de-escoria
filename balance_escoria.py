import flet as ft
import numpy as np
from scipy.optimize import minimize

# -----------------------------------------------------------------------------
# Optimization Engine
# -----------------------------------------------------------------------------


def solve_optimization(carry_mass, carry_chem, materials, target_b2, target_mgo):
    """
    Solves for the mass of each material to add to meet constraints.

    Args:
        carry_mass (float): Mass of carry-over slag.
        carry_chem (dict): Composition of carry-over slag {'Name': %, ...}.
        materials (list): List of dicts [{'name': '...', 'chem': {...}}, ...].
        target_b2 (float): Minimum Basicity (CaO/SiO2).
        target_mgo (float): Minimum %MgO.

    Returns:
        dict: Result containing 'success', 'message', 'added_masses', 'final_chem', 'final_mass'.
    """

    # Oxides of interest
    oxide_list = ["FeO", "CaO", "MgO", "SiO2", "Al2O3", "MnO"]

    # Helper to get mass of a specific oxide from carry-over
    def get_carry_oxide_mass(oxide):
        return carry_mass * (carry_chem.get(oxide, 0.0) / 100.0)

    # Initial guesses (add 1kg of each material)
    n_materials = len(materials)
    x0 = np.ones(n_materials)

    # bounds: x >= 0
    bounds = [(0, None) for _ in range(n_materials)]

    # --- Constraints ---

    # 1. Basicity Constraint: CaO / SiO2 >= Target
    # CaO_total - Target * SiO2_total >= 0
    def constraint_basicity(x):
        mass_cao = get_carry_oxide_mass("CaO")
        mass_sio2 = get_carry_oxide_mass("SiO2")

        for i, mass in enumerate(x):
            mat_chem = materials[i]["chem"]
            mass_cao += mass * (mat_chem.get("CaO", 0.0) / 100.0)
            mass_sio2 += mass * (mat_chem.get("SiO2", 0.0) / 100.0)

        # Avoid division by zero if SiO2 is 0 (unlikely but safe)
        if mass_sio2 == 0:
            return mass_cao  # Treat as infinite basicity if SiO2 is 0

        return mass_cao - (target_b2 * mass_sio2)

    # 2. MgO Constraint: %MgO >= Target
    # MgO_total / Total_Mass >= Target / 100
    # MgO_total - (Target/100) * Total_Mass >= 0
    def constraint_mgo(x):
        total_mass = carry_mass + np.sum(x)
        mass_mgo = get_carry_oxide_mass("MgO")

        for i, mass in enumerate(x):
            mat_chem = materials[i]["chem"]
            mass_mgo += mass * (mat_chem.get("MgO", 0.0) / 100.0)

        return mass_mgo - ((target_mgo / 100.0) * total_mass)

    cons = [
        {"type": "ineq", "fun": constraint_basicity},
        {"type": "ineq", "fun": constraint_mgo},
    ]

    # Objective: Minimize total mass added
    def objective(x):
        return np.sum(x)

    # Optimization
    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=cons, tol=1e-4
    )

    # --- Result Processing ---

    added_masses = {}
    final_chem = {}

    total_mass_final = carry_mass + np.sum(result.x)

    # Calculate final composition
    for oxide in oxide_list:
        mass_oxide = get_carry_oxide_mass(oxide)
        for i, mass in enumerate(result.x):
            mass_oxide += mass * (materials[i]["chem"].get(oxide, 0.0) / 100.0)

        final_chem[oxide] = (
            (mass_oxide / total_mass_final) * 100.0 if total_mass_final > 0 else 0.0
        )

    for i, mat in enumerate(materials):
        added_masses[mat["name"]] = max(
            0.0, result.x[i]
        )  # Ensure no slightly neg floats

    # Calculate final Basicity for report
    final_b2 = final_chem["CaO"] / final_chem["SiO2"] if final_chem["SiO2"] > 0 else 0.0

    return {
        "success": result.success,
        "message": result.message,
        "added_masses": added_masses,
        "final_mass": total_mass_final,
        "final_chem": final_chem,
        "final_b2": final_b2,
    }


# -----------------------------------------------------------------------------
# GUI Components
# -----------------------------------------------------------------------------


class MaterialRow(ft.Container):
    """A row in the material list representing one additive."""

    def __init__(self, remove_callback, name="", defaults=None):
        super().__init__()
        self.remove_callback = remove_callback
        self.padding = 5
        self.height = 60  # Fixed height to prevent layout calculation errors

        self.txt_name = ft.TextField(
            value=name,
            label="Name",
            width=120,
            height=40,
            text_size=12,
            content_padding=5,
        )

        self.inputs = {}
        oxides = ["FeO", "CaO", "MgO", "SiO2", "Al2O3", "MnO"]

        # Default composition if provided
        chem_defaults = defaults if defaults else {}

        row_controls = [self.txt_name]

        for oxide in oxides:
            val = str(chem_defaults.get(oxide, 0.0))
            tf = ft.TextField(
                value=val,
                label=oxide,
                width=70,
                height=40,
                text_size=12,
                content_padding=5,
                keyboard_type=ft.KeyboardType.NUMBER,
            )
            self.inputs[oxide] = tf
            row_controls.append(tf)

        # Delete button
        # Delete button (Workaround: IconButton is incompatible in this env, using Container+Icon)
        btn_del = ft.Container(
            content=ft.Text(
                "X", color=ft.Colors.RED_400, weight=ft.FontWeight.BOLD, size=20
            ),
            on_click=lambda e: self.remove_callback(self),
            tooltip="Remove Material",
            width=40,
            height=40,
            alignment=ft.Alignment(0, 0),
        )
        row_controls.append(btn_del)

        self.content = ft.Row(
            controls=row_controls,
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def get_data(self):
        """Returns valid dict or None if invalid."""
        try:
            chem = {}
            for k, v in self.inputs.items():
                val = float(v.value) if v.value else 0.0
                chem[k] = val
            return {"name": self.txt_name.value, "chem": chem}
        except ValueError:
            return None


def main(page: ft.Page):
    page.title = "Steelmaking Slag Optimization"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20
    page.window_width = 1100
    page.window_height = 800
    page.scroll = ft.ScrollMode.AUTO

    # -------------------------------------------------------------------------
    # State & Utils
    # -------------------------------------------------------------------------

    material_rows_col = ft.Column(
        scroll=ft.ScrollMode.ALWAYS, expand=True
    )  # Scrollable container for rows

    def add_material_row(e=None, name="", defaults=None):
        row = MaterialRow(remove_material_row, name, defaults)
        material_rows_col.controls.append(row)
        material_rows_col.update()

    def remove_material_row(row_instance):
        material_rows_col.controls.remove(row_instance)
        material_rows_col.update()

    # -------------------------------------------------------------------------
    # Input Section: Carry-over Slag & Constraints
    # -------------------------------------------------------------------------

    # Carry-over Inputs
    txt_carry_mass = ft.TextField(label="Mass (kg)", value="1000", width=100)

    carry_inputs = {}
    oxides = ["FeO", "CaO", "MgO", "SiO2", "Al2O3", "MnO"]
    carry_defaults = {"FeO": 15, "CaO": 30, "MgO": 8, "SiO2": 30, "Al2O3": 10, "MnO": 7}

    carry_inputs_row = [txt_carry_mass]
    for oxide in oxides:
        tf = ft.TextField(label=f"%{oxide}", value=str(carry_defaults[oxide]), width=80)
        carry_inputs[oxide] = tf
        carry_inputs_row.append(tf)

    section_carry_over = ft.Container(
        content=ft.Column(
            [
                ft.Text(
                    "Carry-over Slag Parameters", size=16, weight=ft.FontWeight.BOLD
                ),
                ft.Row(carry_inputs_row, wrap=True),
            ]
        ),
        padding=10,
        border=ft.Border.all(1, ft.Colors.WHITE24),
        border_radius=8,
    )

    # Targets
    txt_target_b2 = ft.TextField(label="Target B2 (CaO/SiO2)", value="2.0", width=150)
    txt_target_mgo = ft.TextField(label="Min %MgO", value="10.0", width=150)

    section_targets = ft.Container(
        content=ft.Column(
            [
                ft.Text("Optimization Targets", size=16, weight=ft.FontWeight.BOLD),
                ft.Row([txt_target_b2, txt_target_mgo]),
            ]
        ),
        padding=10,
        border=ft.Border.all(1, ft.Colors.WHITE24),
        border_radius=8,
    )

    # -------------------------------------------------------------------------
    # Materials Section
    # -------------------------------------------------------------------------

    # Pre-populate some defaults
    def populate_defaults(e):
        # Clear existing
        material_rows_col.controls.clear()
        add_material_row(name="Lime", defaults={"CaO": 95, "MgO": 1, "SiO2": 1})
        add_material_row(name="Dolo", defaults={"CaO": 58, "MgO": 38, "SiO2": 1})
        page.update()

    btn_add_mat = ft.FilledButton(
        content=ft.Text("Add Material"), icon="add", on_click=add_material_row
    )
    btn_reset_defaults = ft.TextButton(
        content=ft.Text("Load Defaults"), on_click=populate_defaults
    )

    section_materials = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Text(
                            "Available Materials", size=16, weight=ft.FontWeight.BOLD
                        ),
                        ft.VerticalDivider(width=20),
                        btn_add_mat,
                        btn_reset_defaults,
                    ],
                    alignment=ft.MainAxisAlignment.START,
                ),
                ft.Divider(),
                # Fixed height container for list to avoid rendering bugs
                ft.Container(
                    content=material_rows_col,
                    height=300,  # Fixed height as requested
                    border=ft.Border.all(1, ft.Colors.WHITE10),
                    border_radius=4,
                    padding=5,
                ),
            ]
        ),
        padding=10,
        border=ft.Border.all(1, ft.Colors.WHITE24),
        border_radius=8,
        margin=ft.Margin(top=10, bottom=10),
    )

    # -------------------------------------------------------------------------
    # Results Section
    # -------------------------------------------------------------------------

    txt_result_status = ft.Text("Ready", color=ft.Colors.GREY_400)

    # Grid for results
    results_grid = ft.Column()

    section_results = ft.Container(
        content=ft.Column(
            [
                ft.Text("Optimization Results", size=16, weight=ft.FontWeight.BOLD),
                txt_result_status,
                ft.Divider(),
                results_grid,
            ]
        ),
        padding=10,
        border=ft.Border.all(1, ft.Colors.WHITE24),
        border_radius=8,
        visible=False,
    )

    # -------------------------------------------------------------------------
    # Logic Handling
    # -------------------------------------------------------------------------

    def run_optimization(e):
        try:
            # 1. Parse Carry Over
            c_mass = float(txt_carry_mass.value)
            c_chem = {
                ox: float(input_tf.value) for ox, input_tf in carry_inputs.items()
            }

            # 2. Parse Targets
            t_b2 = float(txt_target_b2.value)
            t_mgo = float(txt_target_mgo.value)

            # 3. Parse Materials
            mats = []
            for ctrl in material_rows_col.controls:
                if isinstance(ctrl, MaterialRow):
                    data = ctrl.get_data()
                    if data:
                        mats.append(data)

            if not mats:
                page.snack_bar = ft.SnackBar(
                    ft.Text("Please add at least one material")
                )
                page.snack_bar.open = True
                page.update()
                return

            # 4. Run Solver
            res = solve_optimization(c_mass, c_chem, mats, t_b2, t_mgo)

            # 5. Display Results
            section_results.visible = True

            if res["success"]:
                txt_result_status.value = f"Optimization Successful: {res['message']}"
                txt_result_status.color = ft.Colors.GREEN_400
            else:
                txt_result_status.value = f"Optimization Failed: {res['message']}"
                txt_result_status.color = ft.Colors.RED_400

            # Formating Output

            # Recipe Table
            recipe_rows = [
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(name)),
                        ft.DataCell(
                            ft.Text(f"{mass:.2f} kg", weight=ft.FontWeight.BOLD)
                        ),
                    ]
                )
                for name, mass in res["added_masses"].items()
                if mass > 0.01  # Show only significant additions
            ]

            recipe_table = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Material")),
                    ft.DataColumn(ft.Text("Mass to Add")),
                ],
                rows=recipe_rows,
                border=ft.border.all(1, ft.Colors.WHITE24),
            )

            # Final Chemistry Columns
            chem_rows = [
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(ox)),
                        ft.DataCell(ft.Text(f"{pct:.2f} %")),
                    ]
                )
                for ox, pct in res["final_chem"].items()
            ]

            # Add Targets to chem table for comparison
            target_summary = f"Targets -> B2: {res['final_b2']:.2f} (Min {t_b2}), %MgO: {res['final_chem']['MgO']:.2f}% (Min {t_mgo})"

            chem_table = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Oxide")),
                    ft.DataColumn(ft.Text("Final %")),
                ],
                rows=chem_rows,
                border=ft.border.all(1, ft.Colors.WHITE24),
            )

            # Update Grid
            results_grid.controls = [
                ft.Text(f"Total Final Mass: {res['final_mass']:.2f} kg", size=14),
                ft.Container(height=10),
                ft.Text("Recipe (Additions):", weight=ft.FontWeight.BOLD),
                recipe_table,
                ft.Container(height=20),
                ft.Text("Final Chemistry:", weight=ft.FontWeight.BOLD),
                ft.Text(target_summary, color=ft.Colors.CYAN_200),
                chem_table,
            ]

            page.update()

        except ValueError as ex:
            page.snack_bar = ft.SnackBar(ft.Text(f"Input Error: {str(ex)}"))
            page.snack_bar.open = True
            page.update()
        except Exception as ex:
            page.snack_bar = ft.SnackBar(ft.Text(f"Error: {str(ex)}"))
            page.snack_bar.open = True
            page.update()

    btn_calc = ft.FilledButton(
        content=ft.Text("Calculate Optimization"),
        icon="calculate",
        on_click=run_optimization,
        height=50,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
    )

    # -------------------------------------------------------------------------
    # Layout Assembly
    # -------------------------------------------------------------------------

    page.add(
        ft.Text(
            "Steelmaking Slag Balance",
            size=24,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.BLUE_200,
        ),
        ft.Divider(),
        section_carry_over,
        section_targets,
        section_materials,
        ft.Divider(),
        ft.Row([btn_calc], alignment=ft.MainAxisAlignment.CENTER),
        ft.Container(height=20),
        section_results,
    )

    # Initialize with formatted defaults
    populate_defaults(None)


if __name__ == "__main__":
    ft.app(target=main)

# -----------------------------------------------------------------------------
# PyInstaller Command
# -----------------------------------------------------------------------------
# To compile this script into a standalone windowed application:
# pyinstaller --noconfirm --onefile --windowed --name "SlagOptimizer" balance_escoria.py
