import flet as ft
import inspect

try:
    print("Flet Version:", ft.__version__)
except:
    print("Flet Version: Unknown (ft.__version__ not found)")

try:
    print("\n--- ft.TextField ---")
    print(inspect.signature(ft.TextField))
except Exception as e:
    print(f"ft.TextField error: {e}")

try:
    print("\n--- ft.Container ---")
    print(inspect.signature(ft.Container))
except Exception as e:
    print(f"ft.Container error: {e}")
