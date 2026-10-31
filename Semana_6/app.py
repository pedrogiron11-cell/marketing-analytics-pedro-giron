import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import silhouette_score, adjusted_rand_score

from kmodes.kprototypes import KPrototypes

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

st.set_page_config(page_title="Segmentación de Clientes", page_icon="📊", layout="wide")

st.title("📊 Segmentación de Clientes")

st.markdown("""
Aplicación para:

- Visualizar segmentos existentes.
- Comparar múltiples técnicas de clustering.
""")


# =====================================================
# SESSION STATE
# =====================================================

if "resultados" not in st.session_state:
    st.session_state.resultados = None

if "resumen" not in st.session_state:
    st.session_state.resumen = None

if "matriz" not in st.session_state:
    st.session_state.matriz = None

if "X_scaled" not in st.session_state:
    st.session_state.X_scaled = None


# =====================================================
# FUNCIONES AUXILIARES
# =====================================================


def detectar_columna_cluster(df):

    candidatos = ["Cluster", "cluster", "grupo", "Grupo", "segmento", "Segmento"]

    for c in candidatos:
        if c in df.columns:
            return c

    return None


def detectar_columna_gasto(df):

    candidatos = ["gasto", "gasto_total", "total_spent", "Spending", "spending"]

    for c in candidatos:
        if c in df.columns:
            return c

    return None


def calcular_silueta(X, labels):

    try:

        labels_unicos = np.unique(labels)

        if len(labels_unicos) <= 1:
            return np.nan

        if len(labels_unicos) >= len(X):
            return np.nan

        return round(silhouette_score(X, labels), 4)

    except:
        return np.nan


# =====================================================
# CARGA DE DATOS
# =====================================================

st.sidebar.header("Carga de datos")

archivo = st.sidebar.file_uploader("Seleccione CSV", type=["csv"])

if archivo is not None:

    df = pd.read_csv(archivo)

else:

    ruta_default = Path("segmentos_notebook.csv")

    if ruta_default.exists():
        df = pd.read_csv(ruta_default)

    else:
        st.warning("No se encontró segmentos_notebook.csv")
        st.stop()


# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs(["📈 Visor de grupos", "⚙️ Comparador"])

# =====================================================
# TAB 1
# =====================================================

with tab1:

    cluster_col = detectar_columna_cluster(df)

    if cluster_col is None:

        st.error("No existe una columna Cluster.")

    else:

        gasto_col = detectar_columna_gasto(df)

        st.subheader("Clientes por grupo")

        conteo = df[cluster_col].value_counts().reset_index()

        conteo.columns = ["Cluster", "Clientes"]

        fig = px.bar(
            conteo, x="Cluster", y="Clientes", color="Cluster", text="Clientes"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Perfil de segmentos")

        perfil = df.groupby(cluster_col).size().reset_index(name="Clientes")

        perfil["% Clientes"] = perfil["Clientes"] / len(df) * 100

        if gasto_col:

            gasto_prom = df.groupby(cluster_col)[gasto_col].mean().reset_index()

            gasto_total = df.groupby(cluster_col)[gasto_col].sum().reset_index()

            total_gasto = df[gasto_col].sum()

            gasto_total["% Gasto"] = gasto_total[gasto_col] / total_gasto * 100

            perfil = perfil.merge(gasto_prom, on=cluster_col)

            perfil = perfil.merge(gasto_total[[cluster_col, "% Gasto"]], on=cluster_col)

        st.dataframe(perfil.round(2), use_container_width=True)


# =====================================================
# TAB 2
# =====================================================

with tab2:

    st.subheader("Comparador de técnicas")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    excluir = ["Cluster", "cluster", "id_cliente", "cliente_id", "id"]

    numeric_cols = [c for c in numeric_cols if c not in excluir]

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    col1, col2 = st.columns(2)

    with col1:

        variables_numericas = st.multiselect(
            "Variables numéricas", numeric_cols, default=numeric_cols[:4]
        )

    with col2:

        variables_categoricas = st.multiselect(
            "Variables categóricas", cat_cols, default=cat_cols[:2]
        )

    n_clusters = st.slider("Número de clusters", 2, 10, 3)

    # ==========================================
    # EJECUTAR
    # ==========================================

    if st.button("Ejecutar comparación"):

        resultados = {}

        X_num = df[variables_numericas].copy()

        X_num = X_num.fillna(X_num.median())

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X_num)

        # ---------------- KMEANS

        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(
            X_scaled
        )

        resultados["KMeans"] = {
            "labels": labels,
            "silhouette": calcular_silueta(X_scaled, labels),
        }

        # ---------------- JERÁRQUICO

        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X_scaled)

        resultados["Jerárquico"] = {
            "labels": labels,
            "silhouette": calcular_silueta(X_scaled, labels),
        }

        # ---------------- DBSCAN

        labels = DBSCAN(eps=0.8, min_samples=5).fit_predict(X_scaled)

        resultados["DBSCAN"] = {
            "labels": labels,
            "silhouette": calcular_silueta(X_scaled, labels),
        }

        # ---------------- GMM

        ruta_gmm = Path("modelo_gmm.joblib")

        try:

            if ruta_gmm.exists():

                obj = joblib.load(ruta_gmm)

                modelo = obj["modelo"]
                scaler_gmm = obj["scaler"]
                vars_gmm = obj["variables"]

                X_gmm = df[vars_gmm].copy()

                X_gmm = scaler_gmm.transform(X_gmm)

                labels = modelo.predict(X_gmm)

            else:

                labels = GaussianMixture(
                    n_components=n_clusters, random_state=42
                ).fit_predict(X_scaled)

            resultados["GMM"] = {
                "labels": labels,
                "silhouette": calcular_silueta(X_scaled, labels),
            }

        except Exception as e:

            st.warning(f"GMM: {e}")

        # ---------------- KPROTO

        if len(variables_categoricas) > 0:

            try:

                X_kproto = df[variables_numericas + variables_categoricas].copy()

                for c in variables_categoricas:
                    X_kproto[c] = X_kproto[c].astype(str).fillna("Desconocido")

                cat_idx = [X_kproto.columns.get_loc(c) for c in variables_categoricas]

                modelo = KPrototypes(
                    n_clusters=n_clusters, init="Huang", random_state=42
                )

                labels = modelo.fit_predict(X_kproto.to_numpy(), categorical=cat_idx)

                resultados["K-Prototypes"] = {"labels": labels, "silhouette": np.nan}

            except Exception as e:

                st.warning(f"K-Prototypes: {e}")

        # ======================================
        # RESUMEN
        # ======================================

        resumen = []

        for metodo, info in resultados.items():

            labels = info["labels"]

            resumen.append(
                {
                    "Técnica": metodo,
                    "Grupos": len(np.unique(labels)),
                    "Tamaños": str(pd.Series(labels).value_counts().to_dict()),
                    "Silhouette": info["silhouette"],
                }
            )

        resumen_df = pd.DataFrame(resumen)

        # ======================================
        # MATRIZ ARI
        # ======================================

        nombres = list(resultados.keys())

        matriz = pd.DataFrame(index=nombres, columns=nombres)

        for m1 in nombres:

            for m2 in nombres:

                matriz.loc[m1, m2] = round(
                    adjusted_rand_score(
                        resultados[m1]["labels"], resultados[m2]["labels"]
                    ),
                    4,
                )

        # ======================================
        # GUARDAR RESULTADOS
        # ======================================

        st.session_state.resultados = resultados

        st.session_state.resumen = resumen_df

        st.session_state.matriz = matriz

        st.session_state.X_scaled = X_scaled

    # ==========================================
    # MOSTRAR RESULTADOS
    # ==========================================

    if st.session_state.resultados:

        st.subheader("Resumen")

        st.dataframe(st.session_state.resumen, use_container_width=True)

        st.subheader("Coincidencia (ARI)")

        st.dataframe(st.session_state.matriz, use_container_width=True)

        resultados = st.session_state.resultados

        X_scaled = st.session_state.X_scaled

        st.subheader("Visualización")

        metodo = st.selectbox("Técnica", list(resultados.keys()), key="selector_metodo")

        labels = resultados[metodo]["labels"]

        plot_df = pd.DataFrame(
            {"x": X_scaled[:, 0], "y": X_scaled[:, 1], "Cluster": labels.astype(str)}
        )

        fig = px.scatter(
            plot_df, x="x", y="y", color="Cluster", title=f"Clusters - {metodo}"
        )

        st.plotly_chart(fig, use_container_width=True)
