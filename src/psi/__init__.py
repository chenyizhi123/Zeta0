from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("zeta0-humanoid")  # distribution name; imports remain psi
except PackageNotFoundError:
    __version__ = "0.2.0"  # fallback if package not installed
