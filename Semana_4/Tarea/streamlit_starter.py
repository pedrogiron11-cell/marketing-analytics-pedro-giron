"""
Laboratorio de Recomendación — PUNTO DE PARTIDA para su propia app de Streamlit.
Complete los `# TODO`.

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus recomendadores:
elegir un cliente y un método, ver sus 3 recomendaciones con el "porque", y comparar los métodos
(hit rate y cobertura) — con al menos un control interactivo.

Correr con:  streamlit run streamlit_starter.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Recomendador — Laboratorio", page_icon="☕", layout="wide"
)

VERDE = "#00704A"
NARANJA = "#C4512F"
GRIS = "#9A988E"


@st.cache_data
def cargar():
    catalogo = pd.read_csv("Semana_4/Tarea/data/catalogo.csv")
    hist = pd.read_csv("Semana_4/Tarea/data/ordenes_historial.csv")
    hold = pd.read_csv("Semana_4/Tarea/data/ordenes_holdout.csv")
    exp = pd.read_csv("Semana_4/Tarea/data/experimento.csv")
    clientes = pd.read_csv("Semana_4/Tarea/data/clientes.csv")
    productos = list(catalogo["item"])
    M = (
        (pd.crosstab(hist["customer_id"], hist["item"]) > 0)
        .astype(int)
        .reindex(index=clientes["customer_id"], columns=productos, fill_value=0)
    )
    X = M.values.astype(np.float32)
    H = (
        (pd.crosstab(hold["customer_id"], hold["item"]) > 0)
        .astype(int)
        .reindex(index=M.index, columns=productos, fill_value=0)
        .values
    )
    NUEVO = H * (1 - X)  # probó algo NUEVO en el holdout
    NUEVO[:, productos.index("Agua")] = 0
    es_control = (
        exp.set_index("customer_id").loc[M.index, "grupo"] == "control"
    ).values
    return productos, M.index.values, X, NUEVO, es_control


productos, ids, X, NUEVO, es_control = cargar()
COL_AGUA = productos.index("Agua")
popularidad = X.sum(axis=0)

# TODO 1: la tabla de similitudes producto × producto (con la diagonal en 0), calculada UNA vez
S_it = cosine_similarity(X.T)
np.fill_diagonal(S_it, 0)


def puntajes(metodo, x_vec, K):
    """Puntajes de los 29 productos para UN cliente (x_vec = su vector de 0/1)."""

    if metodo == "Popularidad":
        return popularidad.astype(float)

    if metodo == "Filtrado por ítem":
        # TODO 2: suma de las similitudes de cada producto con lo que el cliente ya compró
        return x_vec @ S_it

    # TODO 3: filtrado por cliente
    sims = cosine_similarity(x_vec[None, :], X)[0]
    vecinos = np.argsort(-sims)[:K]
    return sims[vecinos] @ X[vecinos]


def top3(scores, x_vec):
    s = np.where(x_vec > 0, -np.inf, scores).astype(
        float
    )  # no recomendar lo ya comprado
    s[COL_AGUA] = -np.inf  # ni el Agua
    return np.argsort(-s)[:3]


# --- Título ---
st.title("☕ Recomendador de productos")
st.caption(
    "TODO: escriban aquí una frase sobre qué decisión de negocio ayuda a tomar esta app."
)

# TODO 4: barra lateral con los controles interactivos (mínimo UNO): un selector del método
#         (st.selectbox) y un slider con el número de vecinos K (st.slider)
with st.sidebar:
    st.header("Parámetros")
    metodo = st.selectbox(
        "Método", ["Popularidad", "Filtrado por ítem", "Filtrado por cliente"]
    )
    K = st.slider(
        "Vecinos (solo para filtrado por cliente)",
        min_value=5,
        max_value=200,
        value=30,
        step=5,
    )

tab1, tab2 = st.tabs(["Recomendar", "Comparar métodos"])

with tab1:
    cid = st.number_input(
        "ID de cliente",
        min_value=int(ids.min()),
        max_value=int(ids.max()),
        value=14,
        step=1,
    )
    fila = int(np.where(ids == cid)[0][0])
    x_vec = X[fila].copy()
    st.write("Ya compró:", [productos[j] for j in np.where(x_vec > 0)[0]])

    # TODO 5: muestren las 3 recomendaciones con su puntaje y, para cada una, el "porque"
    #         - por ítem: "porque compró X (similitud 0.87)"
    #         - por cliente: "N de sus K vecinos más parecidos lo compraron"
    #         - popularidad: "es de lo más comprado"
    #         (y qué hacen con un cliente SIN historial: los métodos colaborativos dan puntaje 0)

    scores = puntajes(metodo, x_vec, K)
    tk = top3(scores, x_vec)

    st.subheader("Recomendaciones")
    for j in tk:
        prod = productos[j]
        punt = scores[j]

        if metodo == "Popularidad":
            porque = "Es de lo más comprado."
        elif metodo == "Filtrado por ítem":
            comprados = np.where(x_vec > 0)[0]
            sims_prod = S_it[j, comprados]
            if len(sims_prod) == 0:
                porque = "Cliente sin historial (puntaje 0)."
            else:
                idx_max = comprados[np.argmax(sims_prod)]
                porque = f"Porque compró {productos[idx_max]} (similitud {sims_prod.max():.2f})."
        else:  # filtrado por cliente
            sims = cosine_similarity(x_vec[None, :], X)[0]
            vecinos = np.argsort(-sims)[:K]
            n_lo_compraron = X[vecinos, j].sum()
            porque = f"{n_lo_compraron} de sus {K} vecinos más parecidos lo compraron."

        st.write(f"**{prod}** — puntaje {punt:.3f}")
        st.caption(porque)


with tab2:
    # TODO 6: tabla de comparación (hit rate@3 y cobertura de cada método) calculada sobre los clientes del grupo
    #         CONTROL que probaron algo nuevo (NUEVO.sum(axis=1) > 0 y es_control), con al menos un control
    #         interactivo (por ejemplo, un slider con cuántos clientes evaluar o el K del filtrado por cliente).
    #         Pista: es la misma lógica de hit_rate_at_k del notebook.
    # st.info("Reemplacen cada TODO con su propio código. Revisen el notebook del laboratorio para la lógica exacta de cada pieza.")
    st.subheader("Comparación de métodos")

    K_tab = st.slider("Vecinos para filtrado por cliente (comparación)", 5, 200, 30, 5)

    def hit_rate(scores):
        top = np.argsort(-scores, axis=1)[:, :3]
        aciertos = [(NUEVO[u, top[u]] > 0).any() for u in range(len(X))]
        mask = es_control & (NUEVO.sum(axis=1) > 0)
        return np.mean(np.array(aciertos)[mask])

    def cobertura(scores):
        top = np.argsort(-scores, axis=1)[:, :3]
        return len(np.unique(top))

    # --- Métodos con puntajes por cliente ---
    S_pop_mat = np.tile(popularidad.astype(float), (len(X), 1))
    S_item_mat = X @ S_it

    # Filtrado por cliente usando K_tab
    S_user_mat = np.zeros_like(X, dtype=float)
    sims_all = cosine_similarity(X)

    for i in range(len(X)):
        sims = sims_all[i]
        vecinos = np.argsort(-sims)[:K_tab]
        S_user_mat[i] = sims[vecinos] @ X[vecinos]

    metodos_comp = {
        "Popularidad": S_pop_mat,
        "Filtrado por ítem": S_item_mat,
        "Filtrado por cliente": S_user_mat,
    }

    filas = []
    for nombre, S in metodos_comp.items():
        filas.append(
            {
                "Método": nombre,
                "Hit rate@3 (control)": hit_rate(S),
                "Cobertura": cobertura(S),
            }
        )

    df_comp = pd.DataFrame(filas)
    st.dataframe(df_comp.round(3))
