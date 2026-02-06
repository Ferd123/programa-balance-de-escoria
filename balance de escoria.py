import flet as ft
import numpy as np
from scipy.optimize import minimize
import pandas as pd

# --- CONSTANTES ---
OXIDOS = ["FeO", "CaO", "MgO", "SiO2", "Al2O3", "MnO"]


class MaterialRow:
    """Fila de material (Fundente/Aditivo)"""

    def __init__(self, remove_callback, index):
        self.remove_callback = remove_callback
        self.index = index

        # Campos
        self.txt_name = ft.TextField(
            value=f"Material {index+1}",
            width=150,
            label="Nombre",
            text_size=12,
            height=40,
        )
        self.inputs = {}
        for ox in OXIDOS:
            val = "0.0"
            if index == 0 and ox == "CaO":
                val = "90.0"
            if index == 1 and ox == "MgO":
                val = "30.0"
            if index == 1 and ox == "CaO":
                val = "55.0"

            self.inputs[ox] = ft.TextField(
                value=val,
                width=60,
                label=f"%{ox}",
                text_size=12,
                height=40,
                content_padding=5,
            )

    @property
    def view(self):
        # CORRECCIÓN CLAVE: Usamos 'content' con un ft.Icon explícito.
        # Esto soluciona el error "IconButton must have either icon..."
        return ft.Row(
            controls=[self.txt_name]
            + list(self.inputs.values())
            + [
                ft.IconButton(
                    content=ft.Icon("delete_outline", color="red400"),
                    on_click=lambda _: self.remove_callback(self),
                )
            ],
            alignment=ft.MainAxisAlignment.START,
            spacing=5,
        )

    def get_data(self):
        try:
            composition = [
                float(self.inputs[ox].value) if self.inputs[ox].value else 0.0
                for ox in OXIDOS
            ]
            return {"name": self.txt_name.value, "composition": np.array(composition)}
        except ValueError:
            return None


def main(page: ft.Page):
    page.title = "Solver de Escorias Siderúrgicas v1.3"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_width = 1250
    page.window_height = 900
    page.padding = 20
    page.scroll = ft.ScrollMode.AUTO

    # --- ESTADO ---
    material_rows = []

    # --- 1. CARRY OVER ---
    st_carry_mass = ft.TextField(
        label="Masa Carry-Over (kg)", value="500", width=150, text_align="right"
    )
    inputs_carry = {
        ox: ft.TextField(
            label=f"%{ox}", value="10.0" if ox == "FeO" else "0.0", width=60
        )
        for ox in OXIDOS
    }

    section_carry = ft.Container(
        content=ft.Column(
            [
                ft.Text(
                    "1. Escoria Remanente (Carry-Over)",
                    size=18,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.BLUE_200,
                ),
                ft.Row([st_carry_mass] + list(inputs_carry.values()), spacing=5),
            ]
        ),
        # CORRECCIÓN: ft.Border (Mayúscula) y quitamos bgcolor para evitar cuadro gris
        padding=15,
        border=ft.Border.all(1, ft.Colors.TRANSPARENT),
        border_radius=10,
    )

    # --- 2. RESTRICCIONES ---
    st_target_b2 = ft.TextField(label="Target B2 (CaO/SiO2)", value="2.5", width=150)
    st_min_mgo = ft.TextField(label="Min %MgO", value="8.0", width=150)

    section_targets = ft.Container(
        content=ft.Column(
            [
                ft.Text(
                    "2. Restricciones del Proceso",
                    size=18,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.GREEN_200,
                ),
                ft.Row(
                    [
                        st_target_b2,
                        st_min_mgo,
                        ft.Text("El solver minimizará la masa total.", italic=True),
                    ]
                ),
            ]
        ),
        padding=15,
        border=ft.Border.all(1, ft.Colors.WHITE24),
        border_radius=10,
    )

    # --- 3. MATERIALES ---
    materials_column = ft.Column(scroll=ft.ScrollMode.ALWAYS, height=300, spacing=10)

    def remove_material(row_instance):
        if row_instance in material_rows:
            material_rows.remove(row_instance)
            materials_column.controls.remove(row_instance.view)
            page.update()

    def add_material(e):
        new_row = MaterialRow(remove_material, len(material_rows))
        material_rows.append(new_row)
        materials_column.controls.append(new_row.view)
        page.update()

    add_material(None)
    add_material(None)

    section_materials = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Text(
                            "3. Materiales Disponibles",
                            size=18,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.ORANGE_200,
                        ),
                        # CORRECCIÓN: Usamos FilledButton y un Icon explícito
                        ft.FilledButton(
                            "Agregar Material", icon="add", on_click=add_material
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                ft.Divider(),
                materials_column,
            ]
        ),
        # Fondo transparente para evitar bloque gris, solo borde
        padding=15,
        border=ft.Border.all(1, ft.Colors.WHITE24),
        border_radius=10,
    )

    # --- 4. RESULTADOS ---
    txt_result = ft.Text(
        value="Listo para calcular...", font_family="Consolas", size=14
    )

    def run_optimization(e):
        txt_result.value = "Calculando..."
        page.update()
        try:
            try:
                m_carry = float(st_carry_mass.value)
                comp_carry = np.array([float(inputs_carry[ox].value) for ox in OXIDOS])
                target_b2 = float(st_target_b2.value)
                target_min_mgo = float(st_min_mgo.value)
            except ValueError:
                txt_result.value = "❌ Error: Revisa que todos los campos sean números."
                txt_result.color = "red"
                page.update()
                return

            materials_data = []
            for row in material_rows:
                d = row.get_data()
                if d:
                    materials_data.append(d)

            if not materials_data:
                txt_result.value = "❌ Error: Agrega al menos un material."
                page.update()
                return

            comps_matrix = np.array([m["composition"] for m in materials_data]).T
            n_vars = len(materials_data)

            # MATH
            def mass_balance(x):
                total_mass = m_carry + np.sum(x)
                final_mass_oxides = m_carry * comp_carry + np.dot(comps_matrix, x)
                final_comp = final_mass_oxides / total_mass
                return total_mass, final_comp

            def objective(x):
                return np.sum(x)

            def constraint_b2(x):
                _, final_comp = mass_balance(x)
                if final_comp[3] < 0.001:
                    return 0
                return (final_comp[1] / final_comp[3]) - target_b2

            def constraint_mgo(x):
                _, final_comp = mass_balance(x)
                return final_comp[2] - target_min_mgo

            x0 = np.full(n_vars, 10.0)
            bounds = [(0.0, None) for _ in range(n_vars)]
            cons = [
                {"type": "ineq", "fun": constraint_b2},
                {"type": "ineq", "fun": constraint_mgo},
            ]

            res = minimize(
                objective, x0, method="SLSQP", bounds=bounds, constraints=cons
            )

            if res.success:
                total_mass, final_comp = mass_balance(res.x)
                report = "✅ **OPTIMIZACIÓN EXITOSA**\n\n"
                report += f"Masa Total: {total_mass:.1f} kg | B2: {(final_comp[1]/final_comp[3]):.2f} | MgO: {final_comp[2]:.2f}%\n"
                report += "--------------------------------------------------\n"
                for i, mat in enumerate(materials_data):
                    if res.x[i] > 0.1:
                        report += f"🔹 AÑADIR {mat['name']}:  {res.x[i]:.2f} kg\n"
                report += "\nQuímica Final:\n" + " | ".join(
                    [f"{ox}: {final_comp[i]:.1f}%" for i, ox in enumerate(OXIDOS)]
                )
                txt_result.value = report
                txt_result.color = "green"
            else:
                txt_result.value = f"⚠️ No se encontró solución: {res.message}"
                txt_result.color = "orange"

        except Exception as ex:
            txt_result.value = f"Error crítico: {str(ex)}"
            txt_result.color = "red"
        page.update()

    # CORRECCIÓN: FilledButton (Estándar moderno)
    btn_calc = ft.FilledButton("CALCULAR BALANCE", on_click=run_optimization, height=50)

    # CORRECCIÓN: Margin.only (Mayúscula)
    page.add(
        section_carry,
        ft.Container(height=10),
        section_targets,
        ft.Container(height=10),
        section_materials,
        ft.Container(height=20),
        btn_calc,
        ft.Container(
            content=txt_result,
            padding=15,
            bgcolor=ft.Colors.BLACK45,
            border_radius=10,
            margin=ft.Margin.only(top=10),
        ),
    )


if __name__ == "__main__":
    # Usamos ft.app target=main. Si tu versión insiste en run(), puedes probar ft.run(main),
    # pero ft.app es lo estándar para escritorio.
    ft.app(target=main)
