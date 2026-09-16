"""
Panel de Diagnóstico Técnico - Prototipo de Detección de Somnolencia
------------------------------------------------------------------------
Panel de análisis de datos y desempeño del modelo de Machine Learning,
con estética inspirada en instrumentación de diagnóstico vehicular.

INSTALACIÓN (una sola vez):
    python -m pip install streamlit pandas plotly numpy

EJECUCIÓN:
    python -m streamlit run dashboard_fatiga.py
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

DB_PATH = "eventos_fatiga.db"
DATASET_PATH = "dataset_fatiga.csv"
METRICAS_PATH = "metricas_modelo.json"
MATRIZ_CONFUSION_PATH = "matriz_confusion.png"

st.set_page_config(
    page_title="Panel Técnico | Detección de Somnolencia",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# PALETA Y TIPOGRAFÍA — estética de instrumentación de diagnóstico vehicular
# --------------------------------------------------------------------------

PALETA = {
    "fondo": "#F2F3F5",
    "panel": "#FFFFFF",
    "borde": "#D6D9DE",
    "texto": "#1B222C",
    "texto_mudo": "#6B7280",
    "ambar": "#B5651D",
    "rojo": "#C23B3B",
    "verde": "#2E8B57",
    "azul": "#2C6FAD",
}

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}

    .bloque-cabecera {{
        border-bottom: 1px solid {PALETA['borde']};
        padding-bottom: 14px;
        margin-bottom: 22px;
    }}
    .titulo-modulo {{
        font-size: 13px;
        letter-spacing: 0.06em;
        color: {PALETA['texto_mudo']};
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        display: block;
        margin-bottom: 10px;
    }}
    .valor-monospace {{ font-family: 'IBM Plex Mono', monospace; }}
    .tarjeta-metrica {{
        background-color: {PALETA['panel']};
        border: 1px solid {PALETA['borde']};
        border-radius: 6px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .tarjeta-metrica .etiqueta {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: {PALETA['texto_mudo']};
        letter-spacing: 0.04em;
    }}
    .tarjeta-metrica .valor {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px;
        font-weight: 600;
        margin-top: 4px;
    }}
    .estado-indicador {{
        display: inline-block; width: 8px; height: 8px;
        border-radius: 50%; margin-right: 8px;
    }}
    .panel-tecnico {{
        background-color: {PALETA['panel']};
        border: 1px solid {PALETA['borde']};
        border-radius: 6px;
        padding: 18px 20px;
    }}
    .fila-tecnica {{
        display: flex; justify-content: space-between; padding: 6px 0;
        border-bottom: 1px dashed {PALETA['borde']};
        font-family: 'IBM Plex Mono', monospace; font-size: 13px;
    }}
    .fila-tecnica:last-child {{ border-bottom: none; }}
    .fila-tecnica .clave {{ color: {PALETA['texto_mudo']}; }}
    .fila-tecnica .dato {{ color: {PALETA['texto']}; }}

    .panel-insight {{
        background-color: rgba(44, 111, 173, 0.07);
        border-left: 3px solid {PALETA['azul']};
        border-radius: 4px;
        padding: 14px 18px;
        font-size: 14px;
        line-height: 1.6;
        margin: 14px 0;
        color: {PALETA['texto']};
    }}
    .panel-insight b {{ color: {PALETA['azul']}; }}
    .panel-alerta {{
        background-color: rgba(194, 59, 59, 0.07);
        border-left: 3px solid {PALETA['rojo']};
        border-radius: 4px;
        padding: 14px 18px;
        font-size: 14px;
        line-height: 1.6;
        margin: 14px 0;
        color: {PALETA['texto']};
    }}
    .panel-alerta b {{ color: {PALETA['rojo']}; }}

    [data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; }}
    .stTabs [data-baseweb="tab"] {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; }}
    section[data-testid="stSidebar"] label {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# CARGA DE DATOS
# --------------------------------------------------------------------------

@st.cache_data(ttl=5)
def cargar_eventos():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["id", "conductor_id", "tipo_evento", "valor_ear", "fecha_hora"])
    conexion = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM eventos", conexion)
    conexion.close()
    if not df.empty:
        df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
        df["hora"] = df["fecha_hora"].dt.hour
        df["fecha"] = df["fecha_hora"].dt.date
    return df


@st.cache_data(ttl=5)
def cargar_dataset_entrenamiento():
    if not os.path.exists(DATASET_PATH):
        return pd.DataFrame(columns=["ear", "mar", "duracion_cierre", "etiqueta"])
    return pd.read_csv(DATASET_PATH)


def cargar_metricas():
    if not os.path.exists(METRICAS_PATH):
        return None
    with open(METRICAS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def grafica_base(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=PALETA["texto"],
        font_family="IBM Plex Mono",
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_xaxes(gridcolor=PALETA["borde"], zerolinecolor=PALETA["borde"], linecolor=PALETA["borde"])
    fig.update_yaxes(gridcolor=PALETA["borde"], zerolinecolor=PALETA["borde"], linecolor=PALETA["borde"])
    return fig


COLOR_CLASE = {
    "normal": PALETA["verde"],
    "fatiga": PALETA["ambar"],
    "fatiga_incipiente": PALETA["ambar"],
    "microsueno": PALETA["rojo"],
    "microsueño": PALETA["rojo"],
}

df_eventos = cargar_eventos()
df_dataset = cargar_dataset_entrenamiento()
metricas = cargar_metricas()

# --------------------------------------------------------------------------
# BARRA LATERAL
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ◆ SOMNOLENCIA-AI")
    st.caption("Panel técnico de diagnóstico · v1.0")
    st.markdown("---")

    seccion = st.radio(
        "Módulo",
        ["Resumen operativo", "Análisis exploratorio", "Desempeño del modelo", "Historial de eventos"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<span class="titulo-modulo">Estado del sistema</span>', unsafe_allow_html=True)

    modelo_ok = os.path.exists("modelo_fatiga.pkl")
    color_estado = PALETA["verde"] if modelo_ok else PALETA["rojo"]
    texto_estado = "MODELO ENTRENADO" if modelo_ok else "SIN ENTRENAR"
    st.markdown(
        f'<div class="valor-monospace" style="font-size:12px; margin-top:6px;">'
        f'<span class="estado-indicador" style="background-color:{color_estado};"></span>{texto_estado}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="valor-monospace" style="font-size:12px; margin-top:4px; color:{PALETA["texto_mudo"]};">'
        f'{len(df_eventos)} eventos registrados</div>',
        unsafe_allow_html=True,
    )
    if metricas:
        st.markdown(
            f'<div class="valor-monospace" style="font-size:11px; margin-top:10px; color:{PALETA["texto_mudo"]};">'
            f'Último entrenamiento:<br>{metricas.get("fecha_entrenamiento", "N/D")}</div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# SECCIÓN 1: RESUMEN OPERATIVO
# --------------------------------------------------------------------------

if seccion == "Resumen operativo":
    st.markdown(
        '<div class="bloque-cabecera"><div><h2 style="margin:0;">Resumen Operativo</h2>'
        '<span class="titulo-modulo">Estado en tiempo real del sistema de monitoreo</span></div></div>',
        unsafe_allow_html=True,
    )

    if df_eventos.empty:
        st.info("Aún no hay eventos registrados. Ejecuta 'deteccion_fatiga.py' para comenzar a generar datos.")
    else:
        total_microsuenos = df_eventos["tipo_evento"].isin(["microsueño", "microsueno"]).sum()
        total_fatiga = df_eventos["tipo_evento"].isin(["fatiga_incipiente", "fatiga"]).sum()
        total_eventos = len(df_eventos)
        hoy = datetime.now().date()
        eventos_hoy = len(df_eventos[df_eventos["fecha"] == hoy])

        c1, c2, c3, c4 = st.columns(4)
        for col, etiqueta, valor, color in [
            (c1, "TOTAL EVENTOS", total_eventos, PALETA["azul"]),
            (c2, "MICROSUEÑOS", total_microsuenos, PALETA["rojo"]),
            (c3, "FATIGA INCIPIENTE", total_fatiga, PALETA["ambar"]),
            (c4, "EVENTOS HOY", eventos_hoy, PALETA["verde"]),
        ]:
            col.markdown(
                f'<div class="tarjeta-metrica"><div class="etiqueta">{etiqueta}</div>'
                f'<div class="valor" style="color:{color};">{valor}</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        col_izq, col_der = st.columns([1.3, 1])

        with col_izq:
            st.markdown('<span class="titulo-modulo">Serie temporal de eventos</span>', unsafe_allow_html=True)
            df_tendencia = (
                df_eventos.groupby([df_eventos["fecha_hora"].dt.floor("min"), "tipo_evento"])
                .size().reset_index(name="cantidad")
            )
            df_tendencia["tipo_evento"] = df_tendencia["tipo_evento"].replace({
                "fatiga_incipiente": "Fatiga incipiente", "microsueño": "Microsueño", "microsueno": "Microsueño",
            })
            fig = px.area(
                df_tendencia, x="fecha_hora", y="cantidad", color="tipo_evento",
                labels={"fecha_hora": "Momento", "cantidad": "Eventos", "tipo_evento": "Tipo"},
                color_discrete_map={"Fatiga incipiente": PALETA["ambar"], "Microsueño": PALETA["rojo"]},
            )
            fig.update_traces(line=dict(width=1.5))
            st.plotly_chart(grafica_base(fig), use_container_width=True)

        with col_der:
            st.markdown('<span class="titulo-modulo">Distribución por franja horaria</span>', unsafe_allow_html=True)
            conteo_hora = (
                df_eventos[df_eventos["tipo_evento"].isin(["microsueño", "microsueno"])]["hora"]
                .value_counts().sort_index().reset_index()
            )
            conteo_hora.columns = ["hora", "cantidad"]
            fig2 = px.bar(
                conteo_hora, x="hora", y="cantidad", color_discrete_sequence=[PALETA["rojo"]],
                labels={"hora": "Hora del día", "cantidad": "Microsueños"},
            )
            st.plotly_chart(grafica_base(fig2), use_container_width=True)

        # --- Bloque de interpretación técnica generado a partir de los datos ---
        if total_microsuenos > 0:
            proporcion_microsueno = total_microsuenos / total_eventos
            hora_pico = conteo_hora.loc[conteo_hora["cantidad"].idxmax(), "hora"] if not conteo_hora.empty else None
            texto_insight = (
                f"Del total de eventos registrados, el <b>{proporcion_microsueno:.1%}</b> corresponde a episodios de "
                f"microsueño, el nivel de mayor severidad definido operacionalmente en este proyecto. "
            )
            if hora_pico is not None:
                texto_insight += (
                    f"La franja horaria de las <b>{int(hora_pico):02d}:00</b> concentra la mayor cantidad de episodios, "
                    f"lo cual es consistente con la literatura revisada, que asocia estas franjas con mayor riesgo de fatiga."
                )
            st.markdown(f'<div class="panel-insight">{texto_insight}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# SECCIÓN 2: ANÁLISIS EXPLORATORIO DE DATOS (dataset de entrenamiento)
# --------------------------------------------------------------------------

elif seccion == "Análisis exploratorio":
    st.markdown(
        '<div class="bloque-cabecera"><div><h2 style="margin:0;">Análisis Exploratorio de Datos</h2>'
        '<span class="titulo-modulo">Caracterización estadística del dataset de entrenamiento</span></div></div>',
        unsafe_allow_html=True,
    )

    if df_dataset.empty:
        st.info("Aún no existe 'dataset_fatiga.csv'. Ejecuta 'recolectar_datos.py' primero.")
    else:
        st.markdown('<span class="titulo-modulo">Resumen estadístico por clase</span>', unsafe_allow_html=True)
        resumen = df_dataset.groupby("etiqueta")[["ear", "mar", "duracion_cierre"]].agg(["mean", "std", "min", "max"]).round(3)
        resumen.columns = [f"{col[0]}_{col[1]}" for col in resumen.columns]
        st.dataframe(resumen, use_container_width=True)

        st.write("")
        col_izq, col_der = st.columns(2)

        with col_izq:
            st.markdown('<span class="titulo-modulo">Distribución del EAR por clase</span>', unsafe_allow_html=True)
            fig = px.histogram(
                df_dataset, x="ear", color="etiqueta", barmode="overlay", opacity=0.65,
                color_discrete_map=COLOR_CLASE, nbins=30,
            )
            st.plotly_chart(grafica_base(fig), use_container_width=True)

        with col_der:
            st.markdown('<span class="titulo-modulo">Dispersión EAR vs. MAR</span>', unsafe_allow_html=True)
            fig2 = px.scatter(
                df_dataset, x="ear", y="mar", color="etiqueta",
                color_discrete_map=COLOR_CLASE, opacity=0.75,
            )
            st.plotly_chart(grafica_base(fig2), use_container_width=True)

        st.write("")
        st.markdown('<span class="titulo-modulo">Duración de cierre ocular por clase (diagrama de caja)</span>', unsafe_allow_html=True)
        fig3 = px.box(df_dataset, x="etiqueta", y="duracion_cierre", color="etiqueta", color_discrete_map=COLOR_CLASE)
        st.plotly_chart(grafica_base(fig3), use_container_width=True)

        st.write("")
        st.markdown('<span class="titulo-modulo">Balance de clases en el dataset</span>', unsafe_allow_html=True)
        conteo_clases = df_dataset["etiqueta"].value_counts().reset_index()
        conteo_clases.columns = ["etiqueta", "cantidad"]
        fig4 = px.bar(conteo_clases, x="etiqueta", y="cantidad", color="etiqueta", color_discrete_map=COLOR_CLASE,
                      labels={"etiqueta": "Clase", "cantidad": "Número de muestras"})
        st.plotly_chart(grafica_base(fig4), use_container_width=True)

        # --- Interpretación técnica del análisis exploratorio ---
        ratio_desbalance = conteo_clases["cantidad"].max() / max(conteo_clases["cantidad"].min(), 1)
        ear_normal = df_dataset[df_dataset["etiqueta"] == "normal"]["ear"].mean() if "normal" in df_dataset["etiqueta"].values else None
        ear_microsueno = df_dataset[df_dataset["etiqueta"].isin(["microsueno", "microsueño"])]["ear"].mean() if df_dataset["etiqueta"].isin(["microsueno", "microsueño"]).any() else None

        texto_eda = "<b>Lectura del análisis exploratorio:</b> "
        if ear_normal is not None and ear_microsueno is not None:
            diferencia = ear_normal - ear_microsueno
            texto_eda += (
                f"el EAR promedio en estado normal ({ear_normal:.3f}) es "
                f"{'considerablemente' if diferencia > 0.08 else 'moderadamente'} superior al de microsueño "
                f"({ear_microsueno:.3f}), una diferencia de {diferencia:.3f} puntos que respalda al EAR como "
                f"la característica más discriminante entre ambos estados, consistente con lo reportado en la literatura [5]."
            )
        st.markdown(f'<div class="panel-insight">{texto_eda}</div>', unsafe_allow_html=True)

        if ratio_desbalance > 2:
            st.markdown(
                f'<div class="panel-alerta"><b>Advertencia metodológica:</b> existe un desbalance de clases de '
                f'{ratio_desbalance:.1f}x entre la clase más y menos representada. Esto puede sesgar el modelo hacia '
                f'la clase mayoritaria; se recomienda ampliar la recolección de las clases subrepresentadas antes de '
                f'reportar los resultados finales de validación.</div>',
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------
# SECCIÓN 3: DESEMPEÑO DEL MODELO
# --------------------------------------------------------------------------

elif seccion == "Desempeño del modelo":
    st.markdown(
        '<div class="bloque-cabecera"><div><h2 style="margin:0;">Desempeño del Modelo</h2>'
        '<span class="titulo-modulo">Validación experimental del clasificador Random Forest</span></div></div>',
        unsafe_allow_html=True,
    )

    if metricas is None:
        st.warning("Aún no se ha entrenado el modelo. Ejecuta 'entrenar_modelo.py' primero.")
    else:
        c1, c2, c3 = st.columns(3)
        for col, etiqueta, valor in [
            (c1, "EXACTITUD (ACCURACY)", f"{metricas['exactitud']:.1%}"),
            (c2, "MUESTRAS DE ENTRENAMIENTO", metricas["muestras_entrenamiento"]),
            (c3, "MUESTRAS DE PRUEBA", metricas["muestras_prueba"]),
        ]:
            col.markdown(
                f'<div class="tarjeta-metrica"><div class="etiqueta">{etiqueta}</div>'
                f'<div class="valor">{valor}</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        col_izq, col_der = st.columns([1.2, 1])

        with col_izq:
            st.markdown('<span class="titulo-modulo">Desempeño por clase</span>', unsafe_allow_html=True)
            filas = []
            for clase, valores in metricas["reporte_por_clase"].items():
                if isinstance(valores, dict) and clase not in ["accuracy"]:
                    filas.append({
                        "Clase": clase,
                        "Precisión": valores.get("precision", 0),
                        "Sensibilidad": valores.get("recall", 0),
                        "F1-score": valores.get("f1-score", 0),
                        "Muestras": int(valores.get("support", 0)),
                    })
            df_reporte = pd.DataFrame(filas)
            st.dataframe(
                df_reporte.style.format({"Precisión": "{:.1%}", "Sensibilidad": "{:.1%}", "F1-score": "{:.1%}"}),
                use_container_width=True, hide_index=True,
            )

            st.markdown('<span class="titulo-modulo">Comparación de métricas por clase</span>', unsafe_allow_html=True)
            df_melt = df_reporte.melt(id_vars="Clase", value_vars=["Precisión", "Sensibilidad", "F1-score"],
                                        var_name="Métrica", value_name="Valor")
            fig = px.bar(df_melt, x="Clase", y="Valor", color="Métrica", barmode="group",
                         color_discrete_sequence=[PALETA["azul"], PALETA["ambar"], PALETA["verde"]])
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(grafica_base(fig), use_container_width=True)

        with col_der:
            st.markdown('<span class="titulo-modulo">Ficha técnica del modelo</span>', unsafe_allow_html=True)
            hp = metricas.get("hiperparametros", {})
            filas_html = "".join([
                f'<div class="fila-tecnica"><span class="clave">{clave}</span><span class="dato">{valor}</span></div>'
                for clave, valor in hp.items()
            ])
            st.markdown(f'<div class="panel-tecnico">{filas_html}</div>', unsafe_allow_html=True)

            st.write("")
            st.markdown('<span class="titulo-modulo">Importancia de características</span>', unsafe_allow_html=True)
            df_importancia = pd.DataFrame(
                list(metricas["importancia_caracteristicas"].items()), columns=["Característica", "Importancia"]
            ).sort_values("Importancia", ascending=True)
            fig_imp = px.bar(df_importancia, x="Importancia", y="Característica", orientation="h",
                              color_discrete_sequence=[PALETA["azul"]])
            fig_imp.update_xaxes(tickformat=".0%")
            st.plotly_chart(grafica_base(fig_imp), use_container_width=True)

        st.write("")
        st.markdown('<span class="titulo-modulo">Matriz de confusión</span>', unsafe_allow_html=True)
        if os.path.exists(MATRIZ_CONFUSION_PATH):
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                st.image(MATRIZ_CONFUSION_PATH, use_container_width=True)
        else:
            st.info("Aún no se ha generado la matriz de confusión.")

        # --- Veredicto técnico automático sobre el modelo ---
        exactitud = metricas["exactitud"]
        clase_mas_debil = min(
            [(c, v.get("f1-score", 1)) for c, v in metricas["reporte_por_clase"].items() if isinstance(v, dict)],
            key=lambda x: x[1], default=(None, None)
        )
        texto_veredicto = (
            f"<b>Interpretación de resultados:</b> el modelo alcanza una exactitud global de <b>{exactitud:.1%}</b> "
            f"sobre el conjunto de prueba. "
        )
        if clase_mas_debil[0] is not None:
            texto_veredicto += (
                f"La clase con menor F1-score es <b>{clase_mas_debil[0]}</b> ({clase_mas_debil[1]:.1%}), lo que "
                f"sugiere que es la más difícil de distinguir para el modelo, probablemente por su similitud "
                f"fisiológica con estados adyacentes en el espectro de la somnolencia."
            )
        if metricas["muestras_prueba"] < 30:
            texto_veredicto += (
                " Dado el tamaño reducido del conjunto de prueba, estos resultados deben interpretarse como "
                "preliminares; se recomienda ampliar el dataset antes de reportar cifras definitivas."
            )
        st.markdown(f'<div class="panel-insight">{texto_veredicto}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# SECCIÓN 4: HISTORIAL DE EVENTOS
# --------------------------------------------------------------------------

elif seccion == "Historial de eventos":
    st.markdown(
        '<div class="bloque-cabecera"><div><h2 style="margin:0;">Historial de Eventos</h2>'
        '<span class="titulo-modulo">Registro completo de detecciones del sistema</span></div></div>',
        unsafe_allow_html=True,
    )

    if df_eventos.empty:
        st.info("No hay eventos registrados todavía.")
    else:
        filtro_tipo = st.multiselect(
            "Filtrar por tipo de evento",
            options=df_eventos["tipo_evento"].unique(),
            default=list(df_eventos["tipo_evento"].unique()),
        )
        df_filtrado = df_eventos[df_eventos["tipo_evento"].isin(filtro_tipo)]
        st.dataframe(
            df_filtrado[["fecha_hora", "conductor_id", "tipo_evento", "valor_ear"]]
            .sort_values("fecha_hora", ascending=False),
            use_container_width=True, hide_index=True,
        )
        st.download_button(
            "Descargar historial (.csv)",
            df_filtrado.to_csv(index=False).encode("utf-8"),
            "historial_eventos.csv", "text/csv",
        )

st.markdown("---")
st.caption("Actualización automática cada 5 s · Recarga el navegador para forzar refresco inmediato")