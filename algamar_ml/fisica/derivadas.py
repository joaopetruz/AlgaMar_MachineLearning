"""Features derivadas do campo de velocidade em grade regular lat/lon (diferenças centradas).

dx = R·cos(φ)·dλ, dy = R·dφ (R = 6 371 km). Com u, v em m/s:
- vorticidade ζ = ∂v/∂x − ∂u/∂y
- divergência δ = ∂u/∂x + ∂v/∂y
- Okubo-Weiss W = sn² + ss² − ζ², sn = ∂u/∂x − ∂v/∂y, ss = ∂v/∂x + ∂u/∂y
- energia cinética = 0,5·(u² + v²)
- direção para onde vai a corrente (0° = norte, 90° = leste), entregue como sin/cos
"""
import numpy as np

R = 6_371_000.0


def _passos(lat, lon):
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    dy = R * np.radians(np.gradient(lat))                  # [ny]
    dx = R * np.cos(np.radians(lat))[:, None] * np.radians(np.gradient(lon))[None, :]   # [ny, nx]
    return dx, dy[:, None]


def gradientes(campo, lat, lon):
    """(∂/∂x, ∂/∂y) de um campo [..., ny, nx]."""
    dx, dy = _passos(lat, lon)
    d_lat = np.gradient(campo, axis=-2) / dy
    d_lon = np.gradient(campo, axis=-1) / dx
    return d_lon, d_lat


def deriva(u, v, lat, lon, temperatura=None):
    """Dict de campos derivados com a mesma forma de u e v."""
    dudx, dudy = gradientes(u, lat, lon)
    dvdx, dvdy = gradientes(v, lat, lon)
    zeta = dvdx - dudy
    sn, ss = dudx - dvdy, dvdx + dudy
    vel = np.hypot(u, v)
    direcao = np.arctan2(u, v)  # rad a partir do norte, sentido horário
    with np.errstate(invalid="ignore"):
        direcao_sin = np.where(vel > 0, np.sin(direcao), np.nan)
        direcao_cos = np.where(vel > 0, np.cos(direcao), np.nan)
    saida = {"velocidade": vel, "direcao_sin": direcao_sin, "direcao_cos": direcao_cos,
             "energia_cinetica": 0.5 * (u ** 2 + v ** 2), "vorticidade": zeta, "divergencia": dudx + dvdy,
             "okubo_weiss": sn ** 2 + ss ** 2 - zeta ** 2}
    if temperatura is not None:
        gx, gy = gradientes(temperatura, lat, lon)
        saida["gradiente_temperatura"] = np.hypot(gx, gy) * 1000.0  # °C/km
    return saida
