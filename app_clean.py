import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import gzip
import json
from typing import Optional, Tuple, List

# Importar utilidades
from utils import (
    load_and_sample_data,
    map_magtype_to_origin,
    optimize_dtypes,
    USE_COLUMNS,
    VOLCANIC_TYPES,
    TECTONIC_TYPES
)

# ==================== CONFIGURACIÓN ====================
st.set_page_config(
    page_title="Dashboard Interactivo de Sismos",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rutas de archivos
CSV_PATH = "data.csv"
PARQUET_PATH = "data/sample_parquet.parquet"

# Paletas de color según el prompt
# Paletas de color según el prompt
COLOR_PALETTES = {
    'volcanic': {
        'start': '#FFC0C0',  # Rojo muy claro (bajo)
        'end': '#8B0000',    # Rojo fuerte (alto)
        'colors': ['#FFC0C0', '#FFB6C1', '#FF6347', '#DC143C', '#A52A2A', '#8B0000']
    },
    'tectonic': {
        'start': '#C0D9FF',  # Azul muy claro (bajo)
        'end': '#00008B',    # Azul fuerte (alto)
        'colors': ['#C0D9FF', '#87CEEB', '#6495ED', '#4169E1', '#0000CD', '#00008B']
    },
    'unknown': {
        'start': '#d9d9d9',  # Gris claro
        'end': '#252525',    # Gris oscuro
        'colors': ['#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525']
    }
}

# Mapeo de nombres a Español
CATEGORY_NAMES = {
    'volcanic': 'Volcánico',
    'tectonic': 'Tectónico',
    'unknown': 'Desconocido'
}

# ==================== FUNCIONES DE CARGA ====================

@st.cache_data(show_spinner=True)
def load_data(force_reload: bool = False, use_full: bool = False, n_samples: int = 400_000) -> pd.DataFrame:
    """
    Carga datos con muestreo estratificado o dataset completo.
    
    Args:
        force_reload: Si True, regenera el Parquet
        use_full: Si True, intenta cargar todo el dataset (riesgo de memoria)
        n_samples: Número aproximado de muestras si no se usa use_full
        
    Returns:
        DataFrame con datos cargados
    """
    if use_full:
        st.warning("⚠️ Modo 'Usar todo' activado. Esto puede requerir mucha memoria.")
        # Para modo completo, cargar sin muestreo (pero con optimizaciones)
        from utils import load_csv_chunks, optimize_dtypes, create_decade_column
        df = load_csv_chunks(CSV_PATH, chunksize=200_000)
        df['origin_type'] = map_magtype_to_origin(df['magType'])
        df = optimize_dtypes(df)
        return df
    else:
        # Construir nombre de archivo dinámico basado en el tamaño de muestra
        parquet_path = f"data/sample_{n_samples}.parquet"
        return load_and_sample_data(
            CSV_PATH,
            parquet_path,
            min_samples=n_samples,
            chunksize=200_000,
            force_reload=force_reload
        )


# ==================== FUNCIONES DE FILTRADO ====================

def filter_data(
    df: pd.DataFrame,
    decades: List[int],
    magtypes: List[str],
    origins: List[str],
    mag_range: Tuple[float, float],
    depth_range: Tuple[float, float]
) -> pd.DataFrame:
    """
    Filtra datos según criterios seleccionados.
    
    Args:
        df: DataFrame original
        decades: Lista de décadas seleccionadas
        magtypes: Lista de tipos de magnitud seleccionados
        origins: Lista de categorías de origen seleccionadas
        mag_range: Tupla (min, max) para rango de magnitud
        depth_range: Tupla (min, max) para rango de profundidad
        
    Returns:
        DataFrame filtrado
    """
    df_filtered = df.copy()
    
    # Filtrar por década
    if decades:
        df_filtered = df_filtered[df_filtered['decade'].isin(decades)]
    
    # Filtrar por categoría de origen
    if origins:
        df_filtered = df_filtered[df_filtered['origin_type'].isin(origins)]
    
    # Filtrar por tipo de magnitud
    if magtypes:
        df_filtered = df_filtered[df_filtered['magType'].isin(magtypes)]
    
    # Filtrar por rango de magnitud
    if df_filtered['mag'].notna().any():
        df_filtered = df_filtered[
            (df_filtered['mag'].notna()) &
            (df_filtered['mag'] >= mag_range[0]) &
            (df_filtered['mag'] <= mag_range[1])
        ]
    
    # Filtrar por rango de profundidad
    if df_filtered['depth'].notna().any():
        df_filtered = df_filtered[
            (df_filtered['depth'].notna()) &
            (df_filtered['depth'] >= depth_range[0]) &
            (df_filtered['depth'] <= depth_range[1])
        ]
    
    return df_filtered


# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def get_color_scale(origins_selected: List[str], magtypes_selected: List[str]) -> dict:
    """
    Determina la paleta de colores según las selecciones.
    
    Args:
        origins_selected: Lista de categorías de origen seleccionadas
        magtypes_selected: Lista de magTypes seleccionados
        
    Returns:
        Diccionario con configuración de colores
    """
    # Determinar qué categorías están presentes
    has_volcanic = any(mt in VOLCANIC_TYPES for mt in magtypes_selected) or 'volcanic' in origins_selected
    has_tectonic = any(mt in TECTONIC_TYPES for mt in magtypes_selected) or 'tectonic' in origins_selected
    
    if has_volcanic and not has_tectonic:
        # Solo volcánico: gradiente rojo
        return {
            'type': 'gradient',
            'palette': COLOR_PALETTES['volcanic'],
            'category': 'volcanic'
        }
    elif has_tectonic and not has_volcanic:
        # Solo tectónico: gradiente azul
        return {
            'type': 'gradient',
            'palette': COLOR_PALETTES['tectonic'],
            'category': 'tectonic'
        }
    else:
        # Mezcla: categórico
        return {
            'type': 'categorical',
            'palette': COLOR_PALETTES,
            'category': 'mixed'
        }


def create_map(df_filtered: pd.DataFrame, origins_selected: List[str], magtypes_selected: List[str]) -> Optional[go.Figure]:
    """
    Crea mapa interactivo con colores según la lógica volcánico/tectónico.
    
    Args:
        df_filtered: DataFrame filtrado
        origins_selected: Categorías de origen seleccionadas
        magtypes_selected: MagTypes seleccionados
        
    Returns:
        Figura de Plotly o None
    """
    if df_filtered.empty:
        return None
    
    df_map = df_filtered.dropna(subset=['latitude', 'longitude']).copy()
    
    if df_map.empty:
        return None
    
    # Obtener configuración de colores
    color_config = get_color_scale(origins_selected, magtypes_selected)
    
    fig = go.Figure()
    
    # Normalizar magnitud globalmente para mantener consistencia de color
    mag_min = df_map['mag'].min()
    mag_max = df_map['mag'].max()
    mag_range = mag_max - mag_min if mag_max > mag_min else 1
    
    # Iterar por cada categoría seleccionada para crear trazas separadas
    # Esto permite que la leyenda muestre los nombres correctos
    
    # Orden: Volcánico, Tectónico, Desconocido
    categories_to_plot = [c for c in ['volcanic', 'tectonic', 'unknown'] 
                         if (c in origins_selected or 
                            (c == 'volcanic' and any(mt in VOLCANIC_TYPES for mt in magtypes_selected)) or
                            (c == 'tectonic' and any(mt in TECTONIC_TYPES for mt in magtypes_selected)))]
    
    for category in categories_to_plot:
        df_cat = df_map[df_map['origin_type'] == category]
        
        if df_cat.empty:
            continue
            
        palette = COLOR_PALETTES[category]
        cat_name = CATEGORY_NAMES.get(category, category.capitalize())
        
        # Calcular colores según magnitud para esta categoría
        colors = []
        for mag in df_cat['mag']:
            if pd.isna(mag):
                colors.append(palette['colors'][0]) # Color base si es NaN
            else:
                # Normalizar entre 0 y 1
                norm = (mag - mag_min) / mag_range
                # Seleccionar color de la paleta (invertido o directo según definición)
                # Aquí asumimos palette['colors'] va de claro a oscuro
                idx = min(int(norm * (len(palette['colors']) - 1)), len(palette['colors']) - 1)
                colors.append(palette['colors'][idx])
        
        # Tamaño de marcadores
        sizes = df_cat['mag'].fillna(3) * 2
        sizes = sizes.clip(lower=2, upper=30)
        
        fig.add_trace(go.Scattermapbox(
            lat=df_cat['latitude'],
            lon=df_cat['longitude'],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.7,
                sizemin=2
            ),
            text=df_cat.apply(
                lambda row: f"<b>{row['place']}</b><br>" +
                           f"Magnitud: {row['mag']:.2f}<br>" +
                           f"Profundidad: {row['depth']:.1f} km<br>" +
                           f"Tipo: {row['magType']}<br>" +
                           f"Categoría: {CATEGORY_NAMES.get(row['origin_type'], row['origin_type'])}<br>" +
                           f"Fecha: {row['time']}",
                axis=1
            ),
            hovertemplate='%{text}<extra></extra>',
            name=cat_name,
            showlegend=True
        ))
    
    fig.update_layout(
        title=f'Mapa Interactivo de Sismos - {len(df_map):,} registros',
        mapbox=dict(
            style='open-street-map',
            center=dict(
                lat=df_map['latitude'].median() if not df_map['latitude'].isna().all() else 0,
                lon=df_map['longitude'].median() if not df_map['longitude'].isna().all() else 0
            ),
            zoom=1
        ),
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 225, 0.95)",  # Amarillo claro (Beige)
            bordercolor="rgba(0, 0, 0, 0.8)",     # Borde negro
            borderwidth=2,
            font=dict(
                color="black",                     # Letras en negro
                size=12
            )
        )
    )
    
    return fig


def create_mag_histogram(df_filtered: pd.DataFrame, facet_by: str = None) -> Optional[go.Figure]:
    """Histograma de magnitud, opcionalmente facetado."""
    if df_filtered.empty:
        return None
    
    if facet_by and facet_by in df_filtered.columns:
        fig = px.histogram(
            df_filtered,
            x='mag',
            nbins=50,
            color='origin_type',
            facet_col=facet_by,
            color_discrete_map={
                'volcanic': COLOR_PALETTES['volcanic']['colors'][2],
                'tectonic': COLOR_PALETTES['tectonic']['colors'][2],
                'unknown': COLOR_PALETTES['unknown']['colors'][2]
            },
            title='Distribución de Magnitudes',
            labels={'mag': 'Magnitud', 'count': 'Cantidad'}
        )
    else:
        fig = px.histogram(
            df_filtered,
            x='mag',
            nbins=50,
            color='origin_type',
            barmode='overlay',
            color_discrete_map={
                'volcanic': COLOR_PALETTES['volcanic']['colors'][2],
                'tectonic': COLOR_PALETTES['tectonic']['colors'][2],
                'unknown': COLOR_PALETTES['unknown']['colors'][2]
            },
            title='Distribución de Magnitudes',
            labels={'mag': 'Magnitud', 'count': 'Cantidad'}
        )
    
    fig.update_layout(height=400)
    return fig


def create_boxplot(df_filtered: pd.DataFrame, group_by: str = 'decade') -> Optional[go.Figure]:
    """Boxplots de magnitud agrupados por categoría."""
    if df_filtered.empty:
        return None
    
    if group_by == 'decade':
        x_col = 'decade'
        title = 'Magnitud por Década'
    else:
        x_col = 'origin_type'
        title = 'Magnitud por Categoría Geodinámica'
    
    fig = px.box(
        df_filtered,
        x=x_col,
        y='mag',
        color='origin_type',
        color_discrete_map={
            'volcanic': COLOR_PALETTES['volcanic']['colors'][2],
            'tectonic': COLOR_PALETTES['tectonic']['colors'][2],
            'unknown': COLOR_PALETTES['unknown']['colors'][2]
        },
        title=title,
        labels={'mag': 'Magnitud'}
    )
    fig.update_layout(height=400)
    return fig


def create_scatter_mag_depth(df_filtered: pd.DataFrame) -> Optional[go.Figure]:
    """Scatter plot de magnitud vs profundidad."""
    if df_filtered.empty:
        return None
    
    # Preparar datos para el scatter
    df_scatter = df_filtered.copy()
    
    # Determinar si usar size basado en gap
    use_gap_size = False
    if 'gap' in df_scatter.columns:
        gap_notna_count = df_scatter['gap'].notna().sum()
        # Usar gap como size solo si hay suficientes valores no nulos (>10%)
        if gap_notna_count > len(df_scatter) * 0.1:
            # Rellenar NaN con un valor por defecto (mediana o 0)
            gap_median = df_scatter['gap'].median()
            if pd.notna(gap_median):
                df_scatter['gap_filled'] = df_scatter['gap'].fillna(gap_median)
            else:
                df_scatter['gap_filled'] = df_scatter['gap'].fillna(0)
            use_gap_size = True
    
    fig = px.scatter(
        df_scatter,
        x='depth',
        y='mag',
        color='origin_type',
        size='gap_filled' if use_gap_size else None,
        hover_data=['place', 'time', 'magType', 'gap'] if 'gap' in df_scatter.columns else ['place', 'time', 'magType'],
        color_discrete_map={
            'volcanic': COLOR_PALETTES['volcanic']['colors'][2],
            'tectonic': COLOR_PALETTES['tectonic']['colors'][2],
            'unknown': COLOR_PALETTES['unknown']['colors'][2]
        },
        title='Magnitud vs Profundidad' + (' (tamaño por GAP)' if use_gap_size else ''),
        labels={'depth': 'Profundidad (km)', 'mag': 'Magnitud'}
    )
    fig.update_layout(height=400)
    return fig


def create_timeseries(df_filtered: pd.DataFrame, show_avg_mag: bool = True) -> Optional[go.Figure]:
    """Serie temporal de sismos con conteo y magnitud promedio."""
    if df_filtered.empty:
        return None
    
    # Determinar granularidad según el rango de fechas
    date_range = (df_filtered['time'].max() - df_filtered['time'].min()).days
    
    if date_range > 3650:  # Más de 10 años: agrupar por año
        df_filtered['period'] = df_filtered['time'].dt.year
        period_label = 'Año'
    else:  # Menos de 10 años: agrupar por mes
        df_filtered['period'] = df_filtered['time'].dt.to_period('M').astype(str)
        period_label = 'Mes'
    
    ts_data = df_filtered.groupby('period').agg({
        'mag': ['count', 'mean']
    }).reset_index()
    ts_data.columns = ['period', 'count', 'avg_mag']
    
    if ts_data.empty:
        return None
    
    # Crear subplot con dos ejes Y
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Conteo de eventos
    fig.add_trace(
        go.Bar(
            x=ts_data['period'],
            y=ts_data['count'],
            name='Cantidad de Eventos',
            marker_color=COLOR_PALETTES['tectonic']['colors'][2],
            opacity=0.7
        ),
        secondary_y=False
    )
    
    # Magnitud promedio (si se solicita)
    if show_avg_mag:
        fig.add_trace(
            go.Scatter(
                x=ts_data['period'],
                y=ts_data['avg_mag'],
                name='Magnitud Promedio',
                line=dict(color=COLOR_PALETTES['volcanic']['colors'][2], width=2),
                mode='lines+markers'
            ),
            secondary_y=True
        )
    
    fig.update_layout(
        height=400,
        title='Serie Temporal de Sismos',
        hovermode='x unified'
    )
    fig.update_xaxes(title_text=f'Período ({period_label})')
    fig.update_yaxes(title_text='Cantidad de Eventos', secondary_y=False)
    if show_avg_mag:
        fig.update_yaxes(title_text='Magnitud Promedio', secondary_y=True)
    
    return fig


def create_gap_distribution(df_filtered: pd.DataFrame) -> Optional[go.Figure]:
    """Histograma o violin plot de distribución de GAP."""
    if df_filtered.empty or 'gap' not in df_filtered.columns:
        return None
    
    df_gap = df_filtered[df_filtered['gap'].notna()].copy()
    
    if df_gap.empty:
        return None
    
    # Usar violin plot para mejor visualización de distribución
    fig = px.violin(
        df_gap,
        y='gap',
        color='origin_type',
        box=True,
        color_discrete_map={
            'volcanic': COLOR_PALETTES['volcanic']['colors'][2],
            'tectonic': COLOR_PALETTES['tectonic']['colors'][2],
            'unknown': COLOR_PALETTES['unknown']['colors'][2]
        },
        title='Distribución de GAP (Brecha Acimutal)',
        labels={'gap': 'GAP (grados)'}
    )
    fig.update_layout(height=400)
    return fig


def create_top_magtype(df_filtered: pd.DataFrame, top_n: int = 10) -> Optional[go.Figure]:
    """Gráfico de barras con top N tipos de magnitud."""
    if df_filtered.empty:
        return None
    
    top_mag = df_filtered['magType'].value_counts().head(top_n).reset_index()
    top_mag.columns = ['magType', 'count']
    
    fig = px.bar(
        top_mag,
        x='count',
        y='magType',
        orientation='h',
        title=f'Top {top_n} Tipos de Magnitud',
        labels={'count': 'Cantidad', 'magType': 'Tipo de Magnitud'},
        color='count',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_origin_type_bar(df_filtered: pd.DataFrame) -> Optional[go.Figure]:
    """Gráfico de barras por categoría de origen."""
    if df_filtered.empty:
        return None
    
    origin_counts = df_filtered['origin_type'].value_counts().reset_index()
    origin_counts.columns = ['origin_type', 'count']
    
    # Mapear colores
    colors = [
        COLOR_PALETTES['volcanic']['colors'][2] if row['origin_type'] == 'volcanic'
        else COLOR_PALETTES['tectonic']['colors'][2] if row['origin_type'] == 'tectonic'
        else COLOR_PALETTES['unknown']['colors'][2]
        for _, row in origin_counts.iterrows()
    ]
    
    fig = px.bar(
        origin_counts,
        x='origin_type',
        y='count',
        title='Conteo por Categoría Geodinámica',
        labels={'count': 'Cantidad', 'origin_type': 'Categoría'},
        color='origin_type',
        color_discrete_map={
            'volcanic': COLOR_PALETTES['volcanic']['colors'][2],
            'tectonic': COLOR_PALETTES['tectonic']['colors'][2],
            'unknown': COLOR_PALETTES['unknown']['colors'][2]
        }
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_pie_chart(df_filtered: pd.DataFrame) -> Optional[go.Figure]:
    """Gráfico de torta de distribución por categoría."""
    if df_filtered.empty:
        return None
    
    # Preparar datos para el gráfico de torta
    counts = df_filtered['origin_type'].value_counts().reset_index()
    counts.columns = ['origin_type', 'count']
    # Mapear nombres a español
    counts['label'] = counts['origin_type'].map(CATEGORY_NAMES)
    
    fig = px.pie(
        counts,
        values='count',
        names='label',
        title='Distribución por Origen Geodinámico',
        color='origin_type', # Usar la clave original para el mapeo de color
        color_discrete_map={
            'volcanic': COLOR_PALETTES['volcanic']['colors'][2],
            'tectonic': COLOR_PALETTES['tectonic']['colors'][2],
            'unknown': COLOR_PALETTES['unknown']['colors'][2]
        },
        hole=0.4
    )
    fig.update_traces(textposition='outside',textinfo='percent+label')
    fig.update_layout(height=400)
    return fig


def create_grouped_bar_chart(df_filtered: pd.DataFrame) -> Optional[go.Figure]:
    """Gráfico de barras agrupadas: Magnitud promedio por década y origen."""
    if df_filtered.empty:
        return None
    
    grouped = df_filtered.groupby(['decade', 'origin_type'])['mag'].mean().reset_index()
    # Mapear nombres a español para la leyenda
    grouped['Categoría'] = grouped['origin_type'].map(CATEGORY_NAMES)
    
    # Calcular promedio global por década
    global_avg = df_filtered.groupby('decade')['mag'].mean().reset_index()
    
    fig = px.bar(
        grouped,
        x='decade',
        y='mag',
        color='origin_type', # Usar clave original para colores
        barmode='group',
        title='Magnitud Promedio por Década y Origen',
        labels={'mag': 'Magnitud Promedio', 'decade': 'Década', 'Categoría': 'Categoría'},
        color_discrete_map={
            'volcanic': COLOR_PALETTES['volcanic']['colors'][2],
            'tectonic': COLOR_PALETTES['tectonic']['colors'][2],
            'unknown': COLOR_PALETTES['unknown']['colors'][2]
        }
    )
    
    # Agregar línea de promedio global
    fig.add_trace(
        go.Scatter(
            x=global_avg['decade'],
            y=global_avg['mag'],
            mode='lines+markers',
            name='Promedio Global',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='Década: %{x}<br>Promedio Global: %{y:.2f}<extra></extra>'
        )
    )
    
    # Actualizar nombres en leyenda
    fig.for_each_trace(lambda t: t.update(name = CATEGORY_NAMES.get(t.name, t.name)))
    fig.update_layout(height=400)
    return fig


def create_freq_polygon(df_filtered: pd.DataFrame) -> Optional[go.Figure]:
    """Polígono de frecuencias de magnitud."""
    if df_filtered.empty:
        return None
    
    # Calcular histograma manualmente para crear líneas
    counts, bins = np.histogram(df_filtered['mag'].dropna(), bins=30)
    mids = 0.5*(bins[1:] + bins[:-1])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mids, 
        y=counts, 
        mode='lines+markers',
        name='Frecuencia',
        line=dict(color='#2E86C1', width=3),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 193, 0.2)'
    ))
    
    fig.update_layout(
        title='Polígono de Frecuencias de Magnitud',
        xaxis_title='Magnitud',
        yaxis_title='Frecuencia',
        height=400
    )
    return fig


def create_decade_counts_chart(df_filtered: pd.DataFrame) -> Optional[go.Figure]:
    """Gráfico de línea con conteo total por década."""
    if df_filtered.empty:
        return None
    
    # Agrupar por década y contar
    counts = df_filtered.groupby('decade').size().reset_index(name='count')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=counts['decade'], 
        y=counts['count'], 
        mode='lines+markers',
        name='Total Registros',
        line=dict(color='#1E88E5', width=3), # Azul 
        fill='tozeroy',
        fillcolor='rgba(30, 136, 229, 0.2)'
    ))
    
    fig.update_layout(
        title='Total de Registros por Década',
        xaxis_title='Década',
        yaxis_title='Cantidad de Eventos',
        height=400
    )
    return fig


# ==================== FUNCIONES DE EXPORTACIÓN ====================

def export_to_csv_gz(df: pd.DataFrame) -> bytes:
    """Exporta DataFrame a CSV comprimido con gzip."""
    csv_str = df.to_csv(index=False)
    return gzip.compress(csv_str.encode('utf-8'))


# ==================== FUNCIONES DE KPIs ====================

def calculate_kpis(df: pd.DataFrame) -> dict:
    """Calcula KPIs del dataset filtrado."""
    kpis = {
        'total_events': len(df),
        'avg_mag': df['mag'].mean() if df['mag'].notna().any() else 0,
        'median_mag': df['mag'].median() if df['mag'].notna().any() else 0,
        'avg_depth': df['depth'].mean() if df['depth'].notna().any() else 0,
        'p95_mag': df['mag'].quantile(0.95) if df['mag'].notna().any() else 0,
        'shallow_pct': (df['depth'] < 10).sum() / len(df) * 100 if df['depth'].notna().any() else 0,
    }
    return kpis


def calculate_detailed_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula medidas de tendencia central, dispersión y posición."""
    if df.empty or 'mag' not in df.columns:
        return pd.DataFrame()
    
    mag = df['mag'].dropna()
    
    stats = {
        'Medida': [
            'Media (Promedio)', 'Mediana (Central)', 'Moda (Más frecuente)',
            'Desviación Estándar', 'Varianza', 'Rango', 'Rango Intercuartil (IQR)',
            'Q1 (25%)', 'Q3 (75%)', 'Percentil 95', 'Percentil 99'
        ],
        'Valor': [
            mag.mean(), mag.median(), mag.mode().iloc[0] if not mag.mode().empty else 0,
            mag.std(), mag.var(), mag.max() - mag.min(), mag.quantile(0.75) - mag.quantile(0.25),
            mag.quantile(0.25), mag.quantile(0.75), mag.quantile(0.95), mag.quantile(0.99)
        ]
    }
    
    return pd.DataFrame(stats)


# ==================== INTERFAZ STREAMLIT ====================

def main():
    """Función principal del dashboard."""
    st.set_page_config(page_title="Análisis Sísmico", page_icon="🌍", layout="wide")
    
    # ==================== SIDEBAR ====================
    with st.sidebar.expander("ℹ️ Acerca del Proyecto", expanded=True):
        st.markdown("""
        **Estudiantes:**
        - 👨‍🎓 Juan David Gelviz
        - 👩‍🎓 Tatiana Castaño Morales
        - 👨‍🎓 William Felipe Rodriguez Gutierrez
        - 👩‍🎓 Victoria Bayona Bernal
        
        **Fuente de Datos:**
        - [USGS Earthquake Hazards Program](https://earthquake.usgs.gov/)
        """)
        st.caption("Desarrollado con Streamlit 🎈")
    st.sidebar.title("🌍 Navegación")
    
    # Inicializar estado de página si no existe
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "dashboard"
        
    # Botones de navegación modernos
    col_nav1, col_nav2 = st.sidebar.columns(2)
    with col_nav1:
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state['current_page'] = "dashboard"
            st.rerun()
    with col_nav2:
        if st.button("🗺️ Mapa", use_container_width=True):
            st.session_state['current_page'] = "map"
            st.rerun()
            
    # Determinar página actual
    page_id = st.session_state['current_page']
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filtros Globales")
    
    # Configuración de Datos
    with st.sidebar.expander("⚙️ Configuración de Datos", expanded=False):
        sample_size = st.slider(
            "Cantidad de muestras",
            min_value=100_000,
            max_value=2_000_000,
            value=400_000,
            step=100_000,
            help="Controla cuántos datos se cargan/muestrean del CSV."
        )
        
        if st.button("Recargar Datos"):
            st.cache_data.clear()
            st.rerun()

    # Cargar datos
    with st.spinner("Cargando datos..."):
        try:
            # Verificar si cambió el tamaño de muestra para forzar recarga si es necesario
            # Nota: load_data está cacheado, si cambian los argumentos se re-ejecuta
            df = load_data(force_reload=False, n_samples=sample_size)
            st.sidebar.caption(f"Datos cargados: {len(df):,}")
        except Exception as e:
            st.error(f"❌ Error al cargar datos: {e}")
            st.stop()
            
    if df.empty:
        st.error("No se pudieron cargar los datos.")
        st.stop()

    # Filtros organizados en Expanders
    
    # 1. Filtros de Tiempo
    with st.sidebar.expander("📅 Tiempo (Décadas)", expanded=True):
        if df['decade'].dtype.name == 'category':
            decades_available = sorted([int(d) for d in df['decade'].cat.categories if pd.notna(d)])
        else:
            decades_available = sorted([int(d) for d in df['decade'].unique() if pd.notna(d)])
        
        # Botón para seleccionar todas
        if st.button("Seleccionar Todas"):
            st.session_state['selected_decades'] = decades_available
            
        # Default: 1980, 1990, 2000 si no hay nada en session state
        if 'selected_decades' not in st.session_state:
            default_decades = [d for d in [1980, 1990, 2000] if d in decades_available]
            if not default_decades and decades_available:
                default_decades = [decades_available[-1]]
            st.session_state['selected_decades'] = default_decades
            
        decades_selected = st.multiselect(
            "Seleccionar décadas",
            decades_available,
            key='selected_decades'
        )
    
    # 2. Filtros Geodinámicos
    with st.sidebar.expander("🌋 Geodinámica", expanded=True):
        # Mapear opciones a español para el usuario
        origins_available = sorted([o for o in df['origin_type'].unique() if pd.notna(o)])
        origin_labels = [CATEGORY_NAMES.get(o, o) for o in origins_available]
        
        origins_selected_labels = st.multiselect(
            "Categoría Geodinámica",
            origin_labels,
            default=origin_labels
        )
        
        # Convertir etiquetas seleccionadas de vuelta a claves internas
        reverse_map = {v: k for k, v in CATEGORY_NAMES.items()}
        origins_selected = [reverse_map.get(l, l) for l in origins_selected_labels]
        
        magtypes_available = sorted([str(m) for m in df['magType'].unique() if pd.notna(m)])
        magtypes_selected = st.multiselect(
            "Tipos de Magnitud",
            magtypes_available,
            default=[]
        )
    
    # 3. Filtros de Métricas
    with st.sidebar.expander("📏 Métricas (Magnitud/Profundidad)", expanded=True):
        # Rango de magnitud (Mínimo -1.0 como solicitado)
        mag_min_data = float(df['mag'].min()) if df['mag'].notna().any() else -1.0
        mag_max_data = float(df['mag'].max()) if df['mag'].notna().any() else 10.0
        
        # Asegurar que el slider empiece al menos en -1.0
        slider_min = min(-1.0, mag_min_data)
        
        mag_range = st.slider(
            "Rango de Magnitud", 
            min_value=slider_min, 
            max_value=mag_max_data, 
            value=(slider_min, mag_max_data)
        )
        
        depth_min_data = float(df['depth'].min()) if df['depth'].notna().any() else 0.0
        depth_max_data = float(df['depth'].max()) if df['depth'].notna().any() else 700.0
        
        depth_range = st.slider(
            "Rango de Profundidad (km)", 
            min_value=depth_min_data, 
            max_value=depth_max_data, 
            value=(depth_min_data, depth_max_data)
        )

    # 4. Acerca del Proyecto
    
        

    # Aplicar filtros
    df_filtered = filter_data(df, decades_selected, magtypes_selected, origins_selected, mag_range, depth_range)
    
    st.sidebar.success(f"✅ Datos: {len(df_filtered):,} / {len(df):,}")
    
    # ==================== PÁGINAS ====================
    
    # ==================== PÁGINAS ====================
    
    if page_id == "dashboard":
        st.title("📊 Dashboard de Análisis Sísmico")
        st.markdown("Análisis estadístico detallado de eventos sísmicos.")
        
        # KPIs
        kpis = calculate_kpis(df_filtered)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Eventos", f"{kpis['total_events']:,}")
        c2.metric("Magnitud Promedio", f"{kpis['avg_mag']:.2f}")
        c3.metric("Profundidad Media", f"{kpis['avg_depth']:.1f} km")
        c4.metric("Sismos Superficiales", f"{kpis['shallow_pct']:.1f}%")
        
        st.markdown("---")
        
        # Sección 1: Distribución y Composición
        st.header("1. Composición y Distribución")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribución por Origen")
            st.plotly_chart(create_pie_chart(df_filtered), use_container_width=True)
            st.info("""
            **¿Qué muestra este gráfico?**
            La proporción porcentual de sismos clasificados como **Volcánicos** vs **Tectónicos** (y Desconocidos).
            
            **¿Cómo se calcula?**
            Se cuenta el número de eventos únicos para cada categoría geodinámica en la muestra filtrada.
            
            **¿Por qué es importante?**
            Permite identificar rápidamente la naturaleza predominante de la actividad sísmica. Un predominio tectónico sugiere actividad de fallas, mientras que un aumento en volcánicos puede indicar actividad magmática.
            """)
            
        with col2:
            st.subheader("Tipos de Magnitud")
            st.plotly_chart(create_top_magtype(df_filtered), use_container_width=True)
            st.info("""
            **¿Qué muestra este gráfico?**
            El conteo de los diferentes métodos de medición de magnitud (ej. Mw, Ml, Md).
            
            **¿Cómo se calcula?**
            Frecuencia de cada código `magType` en los datos.
            
            **¿Por qué es importante?**
            Indica la heterogeneidad de las fuentes de datos. Diferentes escalas (ej. Magnitud de Momento Mw vs Magnitud Local Ml) pueden no ser directamente comparables sin correcciones.
            """)

        st.markdown("---")
        
        # Sección 2: Análisis de Magnitud
        st.header("2. Análisis de Magnitud")
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Polígono de Frecuencias")
            st.plotly_chart(create_freq_polygon(df_filtered), use_container_width=True)
            st.info("""
            **¿Qué muestra este gráfico?**
            La distribución de frecuencias de la variable **Magnitud** usando una línea continua.
            
            **¿Cómo se calcula?**
            Se agrupan las magnitudes en intervalos (bins) y se conectan los puntos medios de cada intervalo.
            
            **¿Por qué es importante?**
            Facilita la visualización de la forma de la distribución (ej. si sigue la ley de Gutenberg-Richter) y permite comparar tendencias mejor que un histograma de barras.
            """)
            
        with col4:
            st.subheader("Magnitud por Década y Origen")
            st.plotly_chart(create_grouped_bar_chart(df_filtered), use_container_width=True)
            st.info("""
            **¿Qué muestra este gráfico?**
            El promedio de la variable **Magnitud** agrupado por década y desglosado por categoría (Volcánico/Tectónico).
            
            **¿Cómo se calcula?**
            Promedio aritmético de `mag` para cada combinación de `decade` y `origin_type`.
            
            **¿Por qué es importante?**
            Permite detectar cambios temporales en la intensidad de los eventos y ver si una categoría específica está volviéndose más energética con el tiempo.
            """)

        st.markdown("---")

        # Sección 2.1: Conteo Total por Década (Nuevo)
        st.header("2.1. Total de Registros por Década")
        fig_decade_counts = create_decade_counts_chart(df_filtered)
        if fig_decade_counts:
            st.plotly_chart(fig_decade_counts, use_container_width=True)
        
        st.info("""
        **¿Qué muestra este gráfico?**
        La cantidad total de eventos sísmicos registrados en cada década, independientemente de su origen.
        
        **¿Cómo se interpreta?**
        Permite visualizar la tendencia temporal en la frecuencia de los sismos. Un aumento podría indicar mayor actividad sísmica o mejoras en la instrumentación y detección.
        """)
        
        st.markdown("---")
        
        # Sección 3: Boxplot (Re-integrado)
        st.header("3. Distribución de Magnitud (Boxplot)")
        fig_box = create_boxplot(df_filtered, group_by='decade')
        if fig_box:
            st.plotly_chart(fig_box, use_container_width=True)
        st.info("""
        **¿Qué muestra este gráfico?**
        La dispersión estadística de la **Magnitud** para cada década.
        
        **¿Cómo se interpreta?**
        - **Caja**: Rango Intercuartil (50% central de los datos).
        - **Línea central**: Mediana.
        - **Bigotes**: Rango total (excluyendo atípicos).
        - **Puntos**: Valores atípicos (outliers).
        
        **¿Por qué es importante?**
        Ayuda a visualizar no solo el promedio, sino la variabilidad y los eventos extremos en cada periodo.
        """)

        st.markdown("---")

        # Sección 4: Estadística Descriptiva
        st.header("4. Estadística Descriptiva Detallada")
        st.markdown("A continuación se presentan las métricas estadísticas calculadas sobre la variable **Magnitud**.")
        stats_df = calculate_detailed_stats(df_filtered)
        
        c_stat1, c_stat2 = st.columns([1, 2])
        
        with c_stat1:
            st.dataframe(stats_df, hide_index=True, height=400)
            
        with c_stat2:
            st.markdown("""
            ### Explicación de Medidas (Variable: Magnitud)
            
            **Medidas de Tendencia Central:**
            - **Media:** El valor promedio de la magnitud. Sensible a valores extremos.
            - **Mediana:** El valor central de la magnitud ordenados. Divide los datos en dos partes iguales. Más robusta ante sismos muy fuertes.
            - **Moda:** El valor de magnitud que más se repite en la muestra.
            
            **Medidas de Dispersión:**
            - **Desviación Estándar:** Cuánto se alejan las magnitudes individuales del promedio.
            - **Varianza:** El cuadrado de la desviación estándar.
            - **Rango:** Diferencia entre el sismo más fuerte y el más débil.
            - **IQR (Rango Intercuartil):** Diferencia entre el tercer y primer cuartil (Q3 - Q1). Mide la dispersión del 50% central de los datos.
            
            **Medidas de Posición:**
            - **Q1 (25%) / Q3 (75%):** Valores de magnitud que dejan por debajo el 25% y 75% de los datos respectivamente.
            - **Percentiles (95, 99):** Indican valores extremos. Por ejemplo, el P95 indica que el 95% de los sismos tienen una magnitud menor a ese valor.
            """)

    elif page_id == "map":
        st.title("🗺️ Mapa Interactivo de Sismos")
        st.markdown(f"Visualización geoespacial de **{len(df_filtered):,}** eventos.")
        
        fig_map = create_map(df_filtered, origins_selected, magtypes_selected)
        if fig_map:
            st.plotly_chart(fig_map, use_container_width=True, height=800)
        else:
            st.warning("No hay datos para mostrar en el mapa.")
            
        st.info("""
        **Guía de Colores:**
        - **Tectónico (Azules):** Gradiente de Blanco (magnitud baja) a Azul Oscuro (magnitud alta).
        - **Volcánico (Rojos):** Gradiente de Blanco (magnitud baja) a Rojo Oscuro (magnitud alta).
        - **Desconocido (Grises):** Gradiente de Blanco a Gris Oscuro.
        
        El **tamaño** de los puntos también es proporcional a la magnitud.
        """)
        
        st.markdown("### Referencias Geológicas")
        col_map1, col_map2 = st.columns(2)
        
        with col_map1:
            st.image("https://ecoexploratorio.org/wp-content/uploads/2022/08/volcanes-en-el-mundo.jpg", 
                     caption="Mapa de Volcanes del Mundo", 
                     use_container_width=True)
            
        with col_map2:
            st.image("https://elordenmundial.com/wp-content/uploads/2020/08/mapa-placas-tectonicas.png", 
                     caption="Mapa de Placas Tectónicas", 
                     use_container_width=True)

        # ==================== REFERENCIAS Y BIBLIOTECAS ====================
        st.markdown("---")
        st.header("📚 Referencias y Recursos")
        
        # Pestaña 1: Base de Datos
        tab1, tab2, tab3 = st.tabs(["🌐 Fuente de Datos", "📦 Bibliotecas Utilizadas", "🔗 Enlaces Útiles"])
        
        with tab1:
            st.subheader("Base de Datos - USGS Earthquake Hazards Program")
            st.markdown("""
            **Fuente Principal:**
            - 🔗 [USGS Earthquake Hazards Program - FDSNWS Event Web Service](https://earthquake.usgs.gov/fdsnws/event/1/query)
            
            **Descripción:**
            La base de datos utilizada en este análisis proviene del **Servicio Web de Eventos FDSNWS (Federally Supported Open Data Network Web Services)** del Servicio Geológico de los Estados Unidos (USGS). Este servicio proporciona acceso a un catálogo comprensivo de eventos sísmicos registrados a nivel mundial desde 1900.
            
            **Características:**
            - ✅ Cobertura global de eventos sísmicos
            - ✅ Datos actualizados regularmente
            - ✅ Acceso gratuito y sin restricciones
            - ✅ Múltiples formatos de salida (GeoJSON, CSV, etc.)
            - ✅ Parámetros de búsqueda flexibles (rango temporal, ubicación, magnitud, etc.)
            
            **URL Base del API:**
            ```
            https://earthquake.usgs.gov/fdsnws/event/1/query
            ```
            
            **Parámetros Utilizados en Este Proyecto:**
            - `format`: geojson (formato de datos)
            - `starttime`: Fecha de inicio (YYYY-MM-DD)
            - `endtime`: Fecha de finalización (YYYY-MM-DD)
            - `limit`: Número máximo de registros por consulta (20,000)
            - `offset`: Desplazamiento para pagination
            - `orderby`: Ordenamiento de resultados (time-asc)
            
            **Referencia Oficial:**
            📖 [Documentación FDSNWS Event Web Service](https://earthquake.usgs.gov/fdsnws/event/1/)
            """)
        
        with tab2:
            st.subheader("Librerías Python Utilizadas")
            
            libraries_data = {
                "Librería": [
                    "Streamlit",
                    "Pandas",
                    "NumPy",
                    "Plotly",
                    "Requests",
                    "tqdm",
                    "Pathlib"
                ],
                "Propósito": [
                    "Framework web interactivo para dashboards",
                    "Manipulación y análisis de datos",
                    "Computación numérica y vectorial",
                    "Visualizaciones interactivas (gráficos, mapas)",
                    "Solicitudes HTTP para descargar datos",
                    "Barra de progreso para descarga de datos",
                    "Manejo de rutas de archivos"
                ],
                "Uso en Proyecto": [
                    "Interfaz completa del dashboard",
                    "Procesamiento de CSV y creación de DataFrames",
                    "Cálculos estadísticos y manipulaciones numéricas",
                    "Mapas, histogramas, scatter plots, boxplots",
                    "Descarga de datos del API de USGS",
                    "Feedback visual durante descargas largas",
                    "Gestión de directorios y archivos"
                ]
            }
            
            st.dataframe(libraries_data, hide_index=True, use_container_width=True)
            
            st.markdown("""
            **Instalación de dependencias:**
            ```bash
            pip install streamlit pandas numpy plotly requests tqdm
            ```
            """)
        
        with tab3:
            st.subheader("Enlaces Útiles y Recursos")
            
            col_link1, col_link2 = st.columns(2)
            
            with col_link1:
                st.markdown("""
                ### Documentación Oficial
                
                - 📘 [Streamlit Docs](https://docs.streamlit.io/)
                - 🐼 [Pandas Documentation](https://pandas.pydata.org/docs/)
                - 🔢 [NumPy Documentation](https://numpy.org/doc/)
                - 📊 [Plotly Python Documentation](https://plotly.com/python/)
                - 🌐 [Requests Library Docs](https://requests.readthedocs.io/)
                """)
            
            with col_link2:
                st.markdown("""
                ### Recursos Sísmicos
                
                - 🌍 [USGS Earthquake Hazards Program](https://earthquake.usgs.gov/)
                - 📡 [USGS Real-time Earthquakes](https://earthquake.usgs.gov/earthquakes/map/)
                - 📖 [Ley de Gutenberg-Richter](https://en.wikipedia.org/wiki/Gutenberg%E2%80%93Richter_law)
                - 🌋 [Smithsonian Volcano Database](https://volcano.si.edu/)
                """)
            
            st.markdown("---")
            st.markdown("""
            ### Cita Recomendada
            
            Si utilizas datos o análisis de este dashboard, se recomienda citar:
            
            > **U.S. Geological Survey (USGS) Earthquake Hazards Program**. 
            > *Earthquake data from the FDSNWS Event Web Service*.
            > Disponible en: https://earthquake.usgs.gov/fdsnws/event/1/query
            
            **Desarrollo del Dashboard:**
            > Gelviz, J. D., Castaño, T., Rodriguez, W. F., & Bayona, B. V. (2025). 
            > *Dashboard Interactivo de Análisis Sísmico*. 
            > Diplomado en Matemáticas - Análisis.
            """)

if __name__ == "__main__":
    main()





