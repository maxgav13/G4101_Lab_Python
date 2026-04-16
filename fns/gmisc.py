import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import Callable, Optional, Dict, Any

def rose_diag_circ(
    angles: np.ndarray, 
    bin_size: int = 20, 
    dir: int = 1,
    density: bool = False, 
    ax: Optional[plt.Axes] = None, 
    **kwargs
    ) -> plt.Axes:
    
    """
    Crea un diagrama de rosas de datos circulares para visualizar la distribución de ángulos.

    Parameters:
    -----------
    angles (array-like): Arreglo de ángulos en grados.
    bin_size (int): Tamaño de los bins para el histograma circular.
    dir (int): Indicador del tipo de datos, 1 para direccionales y 0 para no-direccionales.
    density (bool): Si True, normaliza el histograma para representar densidad.
    ax (matplotlib.axes.Axes, opcional): Eje en el que dibujar. Si None, se crea uno nuevo.
    **kwargs: Argumentos adicionales para plt.bar.

    Returns:
    --------
    matplotlib.axes.Axes: El eje con el diagrama de rosas.
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    angles = np.where(angles > 360, angles - 360, angles)

    x = np.copy(angles)

    if dir == 0:
        y = x + 180  # Agregar 180 grados para datos no-direccionales
        z = np.concatenate([x, y])
    else:
        z = x
    
    z = np.where(z > 360, z - 360, z)

    angles = np.where(dir == 1, angles, angles*2)  # Duplicar ángulos para datos no-direccionales

    angles = np.deg2rad(angles)  # Convertir a radianes

    # Calcular direccion media
    C = np.sum(np.cos(angles))
    S = np.sum(np.sin(angles))
    mean_angle = np.arctan2(S, C)
    mean_angle = np.where(dir == 0, mean_angle/2, mean_angle)  # Ajustar para datos no-direccionales


    # Crear histogramas circulares
    bin_size = 20
    bins = int(360/bin_size)
    e = np.linspace(0,2*np.pi,bins+1)
    b = np.linspace(2*np.pi/(bins*2),2*np.pi-2*np.pi/(bins*2),bins)
    n,e = np.histogram(np.radians(z),bins=e)

    
    # Dibujar el diagrama de rosas
    ax.bar(b, np.sqrt(n), width=(2 * np.pi / bins), **kwargs)
    ax.vlines(mean_angle, 0, np.sqrt(n).max(), colors='r', linestyles='dashed', label='Mean Direction')
    ax.set_xticks(e[0:bins])
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    
    return ax


def wilcoxon_effects(
    x: np.ndarray, 
    y: np.ndarray = None, 
    mu: float = 0
    ):
    
    """
    Calcula tamaños de efecto para la prueba de Wilcoxon.

    Parameters:
    -----------
    x (array-like): Muestra de datos.
    y (array-like, opcional): Segunda muestra para comparación pareada. Si no se proporciona, se asume una prueba de una muestra contra un valor hipotético (mu).
    mu (float, opcional): Valor hipotético para la prueba de una muestra. Ignorado si se proporciona y.

    Returns:
    --------
    dict: Diccionario con los tamaños de efecto calculados y resultados de la prueba de Wilcoxon de scipy.
     - n: número de diferencias no nulas
     - W_plus: suma de rangos positivos
     - W_minus: suma de rangos negativos
     - W_min: mínimo entre W_plus y W_minus
     - expected_W: esperanza de W bajo la hipótesis nula
     - var_W: varianza de W bajo la hipótesis nula
     - Z_approx: estadístico Z aproximado para muestras grandes
     - r: correlación r (tamaño de efecto)
     - r_b (rank_biserial): rank-biserial (tamaño de efecto para pareado o one-sample)
     - CLES: Common Language Effect Size (probabilidad de superioridad)
     - scipy_wilcoxon_stat: estadístico W de la prueba de Wilcoxon de scipy
     - scipy_wilcoxon_p: valor-p de la prueba de Wilcoxon de scipy

    """
    x = np.asarray(x)
    if y is None:
        d = x - mu
    else:
        y = np.asarray(y)
        d = x - y

    # eliminar ceros
    mask = (d != 0)
    d = d[mask]
    n = len(d)
    if n == 0:
        raise ValueError("No quedan diferencias (todas cero).")

    ranks = pd.Series(np.abs(d)).rank(method="average").values
    signs = np.sign(d)

    W_plus = ranks[signs > 0].sum()
    W_minus = ranks[signs < 0].sum()
    W_min = min(W_plus, W_minus)

    # esperanza y varianza (sin correcciones por empates)
    expected_W = n * (n + 1) / 4
    var_W = n * (n + 1) * (2 * n + 1) / 24

    # Z aproximado (normal)
    Z = (W_plus - expected_W) / np.sqrt(var_W)

    # tamaños de efecto
    r = abs(Z) / np.sqrt(n)                      # correlación r
    r_b = 2 * (W_plus - W_minus) / (n * (n + 1))  # rank-biserial (pareado / one-sample)
    cles = 2 * max(W_plus, W_minus) / (n * (n + 1))  # CLES (probabilidad de superioridad)

    # resultado de scipy (estadístico W y p-value)
    try:
        scipy_res = stats.wilcoxon(d)
        scipy_stat, scipy_p = float(scipy_res.statistic), float(scipy_res.pvalue)
    except Exception:
        scipy_stat, scipy_p = None, None

    return pd.Series({
        "n": n,
        "W_plus": W_plus,
        "W_minus": W_minus,
        "W_min": W_min,
        "expected_W": expected_W,
        "var_W": var_W,
        "Z_approx": Z,
        "r": r,
        "r_b (rank_biserial)": r_b,
        "CLES": cles,
        "scipy_wilcoxon_stat": scipy_stat,
        "scipy_wilcoxon_p": scipy_p
    })

def kruskal_effects(
    data: pd.DataFrame = None, 
    dv: str = None, 
    group: str = None, 
    groups_list: list = None
    ):

    """
    Calcula H (Kruskal-Wallis) y tamaños de efecto epsilon^2 y eta_H^2.

    Parameters:
    -----------
    data: DataFrame con los datos
    dv: nombre de la columna de la variable dependiente
    group: nombre de la columna del factor de agrupación
    groups_list: lista de arrays/iterables (alternativa a data)

    Returns:
    --------
    dict: Diccionario con H, p, N, k, epsilon2, eta_H2.
        - H: estadístico H de Kruskal-Wallis
        - p: valor-p de la prueba de Kruskal-Wallis
        - N: número total de observaciones
        - k: número de grupos
        - epsilon2: tamaño de efecto epsilon^2
        - eta_H2: tamaño de efecto eta_H^2

    """
    if groups_list is None:
        if data is None or dv is None or group is None:
            raise ValueError("Proporcione data+dv+group o groups_list")
        grouped = data.groupby(group)[dv].apply(list)
        groups = [np.asarray(g) for g in grouped.values]
    else:
        groups = [np.asarray(g) for g in groups_list]

    # eliminar NA dentro de cada grupo
    groups = [g[~pd.isna(g)] for g in groups]
    k = len(groups)
    N = sum(len(g) for g in groups)
    if k < 2 or N == 0:
        raise ValueError("Se requieren al menos dos grupos con datos.")

    kr = stats.kruskal(*groups)
    H = float(kr.statistic)
    p = float(kr.pvalue)

    # tamaños de efecto
    epsilon2 = H / ((N**2 - 1) / (N + 1)) if N > 1 else np.nan
    eta_H2 = (H - k + 1) / (N - k) if (N - k) > 0 else np.nan
    if not np.isnan(eta_H2) and eta_H2 < 0:
        eta_H2 = 0.0

    return pd.Series({
        "H": H,
        "p": p,
        "N": N,
        "k": k,
        "epsilon2": epsilon2,
        "eta_H2": eta_H2
    })

def stratified_bootstrap(
    df: pd.DataFrame,
    group_col: str,
    stat_func: Callable[[pd.DataFrame], Any],
    n_boot: int = 1000,
    strata_n: Optional[Dict[Any,int]] = None,
    random_state: Optional[int] = None,
    ci: float = 0.95
    ):

    """
    Stratified bootstrap.

    Parameters:
    -----------
    df: DataFrame completo
    group_col: columna con la etiqueta de estrato
    stat_func: función que recibe un DataFrame (la muestra completa) y devuelve un escalar o dict de estimadores
    n_boot: número de remuestreos
    strata_n: dict opcional {estrato: tamaño_remuestreo}. Si None usa el tamaño observado por estrato.
    random_state: semilla
    ci: nivel de confianza (por ejemplo 0.95)

    Returns:
    --------
    dict: Diccionario con 'distribution' (DataFrame), 'ci' (dict) y 'summary' (DataFrame resumen).
     - distribution: DataFrame con la distribución de los estadísticos de interés en cada remuestreo
     - ci: dict con los intervalos de confianza para cada estadístico (clave: (lower, upper))
     - summary: DataFrame resumen con la estimación puntual, error estándar y límites del intervalo de confianza para cada estadístico

    """
    rng = np.random.default_rng(random_state)
    groups = {name: group.reset_index(drop=True) for name, group in df.groupby(group_col)}
    # determinar tamaños por estrato
    if strata_n is None:
        strata_n = {name: len(g) for name, g in groups.items()}

    results = []
    for _ in range(n_boot):
        parts = []
        for name, g in groups.items():
            n = int(strata_n.get(name, len(g)))
            if n == 0:
                continue
            idx = rng.integers(0, len(g), size=n)
            parts.append(g.iloc[idx])
        sample = pd.concat(parts, ignore_index=True)
        res = stat_func(sample)
        results.append(res)

    # normalizar salida a DataFrame
    if len(results) == 0:
        raise ValueError("No hay resultados de bootstrap.")
    # si stat_func devuelve escalares numéricos
    if np.isscalar(results[0]):
        arr = np.array(results)
        dist_df = pd.DataFrame({"stat": arr})
        lower = np.percentile(arr, (1 - ci)/2*100)
        upper = np.percentile(arr, (1 + ci)/2*100)
        ci_dict = {"stat": (lower, upper)}
        summary = pd.DataFrame({
            "estimate": [arr.mean()],
            "std_error": [arr.std(ddof=1)],
            f"ci_lower_{ci:.2f}": [lower],
            f"ci_upper_{ci:.2f}": [upper]
        })
    else:
        # si devuelve dicts o múltiples estadísticas
        # convertir lista de dict -> DataFrame
        dist_df = pd.DataFrame(results)
        ci_dict = {}
        rows = []
        for col in dist_df.columns:
            arr = dist_df[col].dropna().to_numpy()
            lower = np.percentile(arr, (1 - ci)/2*100)
            upper = np.percentile(arr, (1 + ci)/2*100)
            ci_dict[col] = (lower, upper)
            rows.append({
                "stat": col,
                "estimate": arr.mean(),
                "std_error": arr.std(ddof=1),
                f"ci_lower_{ci:.2f}": lower,
                f"ci_upper_{ci:.2f}": upper
            })
        summary = pd.DataFrame(rows).set_index("stat")

    return {
        "distribution": dist_df,
        "ci": ci_dict,
        "summary": summary
    }

def up_down_runs_test(
    data: np.ndarray
    ):

    """
    Realiza la prueba de corridas arriba/abajo para evaluar la aleatoriedad de una secuencia de datos.

    Parameters:
    -----------
        data (np.array): Arreglo de valores numéricos.

    Returns:
    --------
        dict: Resultados de la prueba, incluyendo el estadístico Z y el valor-p.
    """

    # 1. Convertir los datos a una secuencia de '+' y '-' según el movimiento ascendente/descendente
    runs_sequence = []
    for i in range(1, len(data)):
        if data[i] > data[i-1]:
            runs_sequence.append('+')
        elif data[i] < data[i-1]:
            runs_sequence.append('-')
        # Ignorar empates (valores iguales)
        
    # 2. Contar el número de corridas (cambios de signo)
    num_runs = 1  # Comenzar con al menos una corrida
    for i in range(1, len(runs_sequence)):
        if runs_sequence[i] != runs_sequence[i-1]:
            num_runs += 1
            
    n = len(runs_sequence) # Número total de elementos sin empate
    n1 = runs_sequence.count("+")
    n2 = runs_sequence.count("-")

    # 3. Calcular el número esperado de corridas y la desviación estándar para muestras grandes
    # Fórmulas específicas para la prueba de corridas arriba/abajo:
    expected_runs = ((2 * n1 * n2) / (n1 + n2)) + 1
    std_dev_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1)))

    # 4. Calcular el estadístico Z
    z_statistic = (num_runs - expected_runs) / std_dev_runs

    # 5. Calcular el valor-p (prueba bilateral)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))

    return {
    "z_statistic": z_statistic,
    "p_value": p_value,
    "n": n,
    "num_runs": num_runs,
    "expected_runs": expected_runs,
    "n1": n1,
    "n2": n2
}