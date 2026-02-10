import flet as ft
import inspect

try:
    print("\n--- ft.TextField ---")
    print(inspect.signature(ft.TextField))
except Exception as e:
    print(f"ft.TextField error: {e}")
