import streamlit as st
import pandas as pd
import plotly.express as px

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Dashboard Champions League",
    page_icon="⚽",
    layout="wide"
)

# --- PREPARACIÓN DE DATOS ---

# Datos de probabilidades proporcionados
# Los estructuramos como una lista de diccionarios para facilitar la creación del DataFrame
data = [
    {'Partido': 'Qarabağ vs Chelsea', 'Equipo': 'Qarabağ', 'Prob_Victoria': '18.6%', 'Prob_Mas1_Gol': '11.9%', 'Prob_Mas2_Goles': '4.8%', 'Prob_Mas3_Goles': '1.5%', 'Prob_Mas4_Goles': '0.4%'},
    {'Partido': 'Qarabağ vs Chelsea', 'Equipo': 'Chelsea', 'Prob_Victoria': '60.7%', 'Prob_Mas1_Gol': '23.3%', 'Prob_Mas2_Goles': '18.4%', 'Prob_Mas3_Goles': '10.9%', 'Prob_Mas4_Goles': '5.2%'},
    {'Partido': 'Qarabağ vs Chelsea', 'Equipo': 'Empate', 'Prob_Victoria': '20.6%', 'Prob_Mas1_Gol': '0%', 'Prob_Mas2_Goles': '0%', 'Prob_Mas3_Goles': '0%', 'Prob_Mas4_Goles': '0%'},
    
    {'Partido': 'Inter vs Kairat Almaty', 'Equipo': 'Inter', 'Prob_Victoria': '84.0%', 'Prob_Mas1_Gol': '20.0%', 'Prob_Mas2_Goles': '22.7%', 'Prob_Mas3_Goles': '18.7%', 'Prob_Mas4_Goles': '12.0%'},
    {'Partido': 'Inter vs Kairat Almaty', 'Equipo': 'Kairat Almaty', 'Prob_Victoria': '4.6%', 'Prob_Mas1_Gol': '3.7%', 'Prob_Mas2_Goles': '0.8%', 'Prob_Mas3_Goles': '<0.1%', 'Prob_Mas4_Goles': '0.0%'},
    {'Partido': 'Inter vs Kairat Almaty', 'Equipo': 'Empate', 'Prob_Victoria': '11.5%', 'Prob_Mas1_Gol': '0%', 'Prob_Mas2_Goles': '0%', 'Prob_Mas3_Goles': '0%', 'Prob_Mas4_Goles': '0%'},
    
    {'Partido': 'Man. City vs B. Dortmund', 'Equipo': 'Man. City', 'Prob_Victoria': '57.8%', 'Prob_Mas1_Gol': '17.2%', 'Prob_Mas2_Goles': '17.2%', 'Prob_Mas3_Goles': '10.5%', 'Prob_Mas4_Goles': '5.2%'},
    {'Partido': 'Man. City vs B. Dortmund', 'Equipo': 'B. Dortmund', 'Prob_Victoria': '22.3%', 'Prob_Mas1_Gol': '7.3%', 'Prob_Mas2_Goles': '6.2%', 'Prob_Mas3_Goles': '2.3%', 'Prob_Mas4_Goles': '0.7%'},
    {'Partido': 'Man. City vs B. Dortmund', 'Equipo': 'Empate', 'Prob_Victoria': '19.9%', 'Prob_Mas1_Gol': '0%', 'Prob_Mas2_Goles': '0%', 'Prob_Mas3_Goles': '0%', 'Prob_Mas4_Goles': '0%'},
    
    {'Partido': 'Club Brugge vs FC Barcelona', 'Equipo': 'Club Brugge', 'Prob_Victoria': '19.4%', 'Prob_Mas1_Gol': '11.9%', 'Prob_Mas2_Goles': '5.2%', 'Prob_Mas3_Goles': '1.7%', 'Prob_Mas4_Goles': '0.5%'},
    {'Partido': 'Club Brugge vs FC Barcelona', 'Equipo': 'FC Barcelona', 'Prob_Victoria': '61.0%', 'Prob_Mas1_Gol': '22.2%', 'Prob_Mas2_Goles': '18.2%', 'Prob_Mas3_Goles': '11.4%', 'Prob_Mas4_Goles': '5.7%'},
    {'Partido': 'Club Brugge vs FC Barcelona', 'Equipo': 'Empate', 'Prob_Victoria': '19.6%', 'Prob_Mas1_Gol': '0%', 'Prob_Mas2_Goles': '0%', 'Prob_Mas3_Goles': '0%', 'Prob_Mas4_Goles': '0%'},
    
    {'Partido': 'Pafos vs Villarreal', 'Equipo': 'Pafos', 'Prob_Victoria': '27.0%', 'Prob_Mas1_Gol': '15.0%', 'Prob_Mas2_Goles': '7.0%', 'Prob_Mas3_Goles': '2.5%', 'Prob_Mas4_Goles': '0.6%'},
    {'Partido': 'Pafos vs Villarreal', 'Equipo': 'Villarreal', 'Prob_Victoria': '47.0%', 'Prob_Mas1_Gol': '22.0%', 'Prob_Mas2_Goles': '14.0%', 'Prob_Mas3_Goles': '7.5%', 'Prob_Mas4_Goles': '3.0%'},
    {'Partido': 'Pafos vs Villarreal', 'Equipo': 'Empate', 'Prob_Victoria': '26.0%', 'Prob_Mas1_Gol': '0%', 'Prob_Mas2_Goles': '0%', 'Prob_Mas3_Goles': '0%', 'Prob_Mas4_Goles': '0%'},
    
    {'Partido': 'Newcastle vs Athletic Club', 'Equipo': 'Newcastle', 'Prob_Victoria': '56.0%', 'Prob_Mas1_Gol': '23.0%', 'Prob_Mas2_Goles': '16.0%', 'Prob_Mas3_Goles': '9.0%', 'Prob_Mas4_Goles': '4.0%'},
    {'Partido': 'Newcastle vs Athletic Club', 'Equipo': 'Athletic Club', 'Prob_Victoria': '20.0%', 'Prob_Mas1_Gol': '11.0%', 'Prob_Mas2_Goles': '6.0%', 'Prob_Mas3_Goles': '3.0%', 'Prob_Mas4_Goles': '1.5%'},
    {'Partido': 'Newcastle vs Athletic Club', 'Equipo': 'Empate', 'Prob_Victoria': '24.0%', 'Prob_Mas1_Gol': '0%', 'Prob_Mas2_Goles': '0%', 'Prob_Mas3_Goles': '0%', 'Prob_Mas4_Goles': '0%'},
    
    {'Partido': 'Ajax vs Galatasaray', 'Equipo': 'Ajax', 'Prob_Victoria': '59.0%', 'Prob_Mas1_Gol': '23.0%', 'Prob_Mas2_Goles': '17.0%', 'Prob_Mas3_Goles': '10.0%', 'Prob_Mas4_Goles': '5.0%'},
    {'Partido': 'Ajax vs Galatasaray', 'Equipo': 'Galatasaray', 'Prob_Victoria': '18.0%', 'Prob_Mas1_Gol': '10.0%', 'Prob_Mas2_Goles': '4.5%', 'Prob_Mas3_Goles': '1.5%', 'Prob_Mas4_Goles': '0.4%'},
    {'Partido': 'Ajax vs Galatasaray', 'Equipo': 'Empate', 'Prob_Victoria': '23.0%', 'Prob_Mas1_Gol': '0%', 'Prob_Mas2_Goles': '0%', 'Prob_Mas3_Goles': '0%', 'Prob_Mas4_Goles': '0%'},
    
    {'Partido': 'Benfica vs Leverkusen', 'Equipo': 'Benfica', 'Prob_Victoria': '49.0%', 'Prob_Mas1_Gol': '21.0%', 'Prob_Mas2_Goles': '14.0%', 'Prob_Mas3_Goles': '7.0%', 'Prob_Mas4_Goles': '3.0%'},
    {'Partido': 'Benfica vs Leverkusen', 'Equipo': 'Leverkusen', 'Prob_Victoria': '26.0%', 'Prob_Mas1_Gol': '13.0%', 'Prob_Mas2_Goles': '7.0%', 'Prob_Mas3_Goles': '3.0%', 'Prob_Mas4_Goles': '1.0%'},
    {'Partido': 'Benfica vs Leverkusen', 'Equipo': 'Empate', 'Prob_Victoria': '25.0%', 'Prob_Mas1_Gol': '0%', 'Prob_Mas2_Goles': '0%', 'Prob_Mas3_Goles': '0%', 'Prob_Mas4_Goles': '0%'},
    
    {'Partido': 'Marsella vs Atalanta', 'Equipo': 'Marsella', 'Prob_Victoria': '39.0%', 'Prob_Mas1_Gol': '19.0%', 'Prob_Mas2_Goles': '11.0%', 'Prob_Mas3_Goles': '5.0%', 'Prob_Mas4_Goles': '2.0%'},
    {'Partido': 'Marsella vs Atalanta', 'Equipo': 'Atalanta', 'Prob_Victoria': '34.0%', 'Prob_Mas1_Gol': '17.0%', 'Prob_Mas2_Goles': '9.0%', 'Prob_Mas3_Goles': '4.0%', 'Prob_Mas4_Goles': '1.5%'},
    {'Partido': 'Marsella vs Atalanta', 'Equipo': 'Empate', 'Prob_Victoria': '27.0%', 'Prob_Mas1_Gol': '0%', 'Prob_Mas2_Goles': '0%', 'Prob_Mas3_Goles': '0%', 'Prob_Mas4_Goles': '0%'},
]

# Crear el DataFrame
df = pd.DataFrame(data)

# Función para limpiar y convertir las cadenas de porcentaje a flotantes
def porcentaje_a_float(valor):
    if isinstance(valor, str):
        valor = valor.replace('%', '')
        if '<' in valor:
            # Tratamos valores como '<0.1%' como 0.0005 para fines de visualización
            return 0.0005 
        return float(valor) / 100.0
    return valor

# Lista de columnas de probabilidad
columnas_prob = ['Prob_Victoria', 'Prob_Mas1_Gol', 'Prob_Mas2_Goles', 'Prob_Mas3_Goles', 'Prob_Mas4_Goles']

# Aplicar la conversión a las columnas de probabilidad
for col in columnas_prob:
    df[col] = df[col].apply(porcentaje_a_float)

# Obtener la lista de partidos únicos para el selector
partidos_unicos = df['Partido'].unique()


# --- INTERFAZ DE STREAMLIT ---

st.title("⚽ Dashboard de Predicciones - Champions League")
st.markdown("Análisis de probabilidades para los partidos de la Champions League. Selecciona un partido en el menú de la izquierda para ver los detalles.")

# Selector de partido en la barra lateral
selected_match = st.sidebar.selectbox("Selecciona un Partido:", partidos_unicos)

# Filtrar el DataFrame según el partido seleccionado
filtered_df = df[df['Partido'] == selected_match]

# --- VISUALIZACIONES ---

st.header(f"Análisis del Partido: {selected_match}")

# Gráfico 1: Probabilidad de Victoria (ELO)
st.subheader("1. Probabilidad de Victoria (ELO)")
fig_win = px.bar(
    filtered_df, 
    x='Equipo', 
    y='Prob_Victoria',
    labels={'Prob_Victoria': 'Probabilidad de Victoria', 'Equipo': 'Equipo / Resultado'},
    title='Comparación de Probabilidades de Victoria y Empate',
    text_auto='.1%' # Muestra el valor como porcentaje en la barra
)
fig_win.update_layout(yaxis_tickformat='.1%', xaxis_title="Equipo / Resultado", yaxis_title="Probabilidad")
st.plotly_chart(fig_win, use_container_width=True)

# Gráfico 2: Probabilidad de Diferencia de Goles
st.subheader("2. Probabilidad de Victoria por Diferencia de Goles")

# Preparar datos para el gráfico de goles
# Excluimos el empate y "derritimos" el DataFrame para el formato largo (ideal para plotly)
teams_df = filtered_df[filtered_df['Equipo'] != 'Empate']
df_melted = teams_df.melt(
    id_vars=['Equipo'], 
    value_vars=['Prob_Mas1_Gol', 'Prob_Mas2_Goles', 'Prob_Mas3_Goles', 'Prob_Mas4_Goles'],
    var_name='Diferencia de Goles', 
    value_name='Probabilidad'
)

# Limpiar los nombres de las categorías de goles
df_melted['Diferencia de Goles'] = df_melted['Diferencia de Goles'].str.replace('Prob_Mas', '+').str.replace('_Goles', '')

fig_goals = px.bar(
    df_melted, 
    x='Diferencia de Goles', 
    y='Probabilidad', 
    color='Equipo',
    barmode='group', # Agrupa las barras por equipo
    labels={'Probabilidad': 'Probabilidad de Ganar con esa Diferencia', 'Diferencia de Goles': 'Diferencia de Goles'},
    title='Distribución de Probabilidades de Margen de Victoria',
    text_auto='.1%'
)
fig_goals.update_layout(yaxis_tickformat='.1%', xaxis_title="Diferencia de Goles", yaxis_title="Probabilidad")
st.plotly_chart(fig_goals, use_container_width=True)


# Tabla de datos detallados
st.subheader("3. Tabla de Probabilidades Detalladas")
# Formateamos el DataFrame para mostrarlo como porcentajes
df_display = filtered_df.copy()
for col in columnas_prob:
    df_display[col] = df_display[col].map('{:.1%}'.format)

# Renombrar columnas para mayor claridad
df_display_renamed = df_display.rename(columns={
    'Prob_Victoria': 'Prob. ELO Victoria',
    'Prob_Mas1_Gol': 'Prob. +1 Gol (Diferencia)',
    'Prob_Mas2_Goles': 'Prob. +2 Goles (Diferencia)',
    'Prob_Mas3_Goles': 'Prob. +3 Goles (Diferencia)',
    'Prob_Mas4_Goles': 'Prob. +4 Goles (Diferencia)'
})

st.dataframe(df_display_renamed, use_container_width=True, hide_index=True)
