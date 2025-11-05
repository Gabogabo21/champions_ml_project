import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json

# --- FUNCIONES DE PROCESAMIENTO (Tuyas, con ligeras modificaciones) ---

def clean_percentage(value):
    """Limpia valores de porcentaje y los convierte a decimal"""
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        clean_val = value.replace('%', '').strip()
        if clean_val == 'NaN':
            return 0.0
        try:
            return float(clean_val) / 100.0
        except:
            return 0.0
    try:
        return float(value)
    except:
        return 0.0

def clean_numeric(value):
    """Limpia valores numéricos"""
    if pd.isna(value):
        return 0.0
    try:
        return float(value)
    except:
        return 0.0

@st.cache_data
def process_data():
    """
    Procesa los datos. 
    IMPORTANTE: He simulado la lectura del Excel para que el ejemplo funcione.
    Debes reemplazar el código de simulación por: df = pd.read_excel('datos_champions.xlsx', header=None)
    """
    # --- INICIO DE SIMULACIÓN ---
    # Crea un DataFrame de ejemplo que imite tu Excel. BORRAR ESTA PARTE.
    st.info("ℹ️ **Modo Demostración**: Se están usando datos simulados. Para usar tus datos, sube tu archivo Excel y ajusta la función `process_data`.")
    data = {
        2: [82.5, 90.1, 88.3, 75.2, 93.5, 85.0, 79.8, 91.2], # Ratings
        3: ['5%', '2%', '3%', '8%', '1%', '4%', '6%', '2%'],   # Tilts
        4: [1850, 1950, 1920, 1750, 1980, 1880, 1800, 1960], # ELO General
        5: [1750, 1900, 1890, 1700, 1950, 1850, 1780, 1940], # ELO País
        6: ['65%', '80%', '75%', '55%', '82%', '70%', '60%', '78%'], # Prob Victoria
        7: ['20%', '15%', '18%', '25%', '12%', '20%', '22%', '16%'], # Prob Empate
        8: [2.1, 2.8, 2.5, 1.8, 3.0, 2.3, 1.9, 2.9], # Goles Esperados
        9: [1, 2, 3, 4, 5, 6, 7, 8], # Posición Actual
        10:[3, 6, 5, 0, 9, 4, 1, 7]  # Puntos Actuales
    }
    df = pd.DataFrame(data)
    # --- FIN DE SIMULACIÓN ---
    
    # Descomenta la siguiente línea para usar tu archivo de Excel real
    # df = pd.read_excel('datos_champions.xlsx', header=None)

    team_names = ['Qarabağ', 'Chelsea', 'Inter', 'Kairat Almaty', 'Man. City', 'B. Dortmund', 'Club Brugge', 'FC Barcelona']
    
    # Extraer datos
    ratings = df.iloc[2, :].values
    tilts = df.iloc[3, :].values
    elo_general = df.iloc[4, :].values
    elo_country = df.iloc[5, :].values
    prob_victoria = df.iloc[6, :].values
    prob_empate = df.iloc[7, :].values
    goles_esperados = df.iloc[8, :].values
    posicion_actual = df.iloc[9, :].values
    puntos_actuales = df.iloc[10, :].values
    
    teams_data = {}
    for i, team in enumerate(team_names):
        teams_data[team] = {
            'rating': clean_numeric(ratings[i]),
            'tilt': clean_percentage(tilts[i]),
            'elo_general': int(clean_numeric(elo_general[i])),
            'elo_country': int(clean_numeric(elo_country[i])),
            'prob_victoria': clean_percentage(prob_victoria[i]),
            'prob_empate': clean_percentage(prob_empate[i]),
            'goles_esperados': clean_numeric(goles_esperados[i]),
            'posicion_actual': int(clean_numeric(posicion_actual[i])),
            'puntos_actuales': int(clean_numeric(puntos_actuales[i]))
        }
    
    # Datos de partidos de hoy
    matches_raw = [
        {'home': 'Qarabağ', 'away': 'Chelsea', 'home_prob': 18.6, 'away_prob': 60.7, 'draw_prob': 20.6},
        {'home': 'Inter', 'away': 'Kairat Almaty', 'home_prob': 84.0, 'away_prob': 4.6, 'draw_prob': 11.5},
        {'home': 'Man. City', 'away': 'B. Dortmund', 'home_prob': 57.8, 'away_prob': 22.3, 'draw_prob': 19.9},
        {'home': 'Club Brugge', 'away': 'FC Barcelona', 'home_prob': 19.4, 'away_prob': 61.0, 'draw_prob': 19.6}
    ]
    
    match_data = []
    for match in matches_raw:
        if match['home'] in teams_data and match['away'] in teams_data:
            match_data.append({
                'home_team': match['home'],
                'away_team': match['away'],
                'date': '2024-05-08', # Puse una fecha realista
                'home_win_prob': match['home_prob'] / 100.0,
                'away_win_prob': match['away_prob'] / 100.0,
                'draw_prob': match['draw_prob'] / 100.0,
                'home_rating': teams_data[match['home']]['rating'],
                'away_rating': teams_data[match['away']]['rating'],
                'home_elo': teams_data[match['home']]['elo_general'],
                'away_elo': teams_data[match['away']]['elo_general'],
                'home_goals_xg': teams_data[match['home']]['goles_esperados'],
                'away_goals_xg': teams_data[match['away']]['goles_esperados'],
                'elo_diff': teams_data[match['home']]['elo_general'] - teams_data[match['away']]['elo_general'],
                'rating_diff': teams_data[match['home']]['rating'] - teams_data[match['away']]['rating']
            })
    
    return teams_data, match_data

# --- FUNCIONES DE MACHINE LEARNING ---

@st.cache_resource
def train_prediction_model(_teams_data, _match_data):
    """
    Entrena un modelo de Machine Learning para predecir resultados.
    NOTA: Para un modelo real, necesitarías HISTÓRICO de partidos.
    Aquí, usamos los datos de hoy como si fueran históricos para demostrar el flujo.
    """
    st.warning("⚠️ **Aviso Importante**: Este modelo es una demostración. Se está entrenando con datos de hoy, lo cual no es correcto. Para un modelo preciso, necesitarías un archivo CSV con cientos de partidos históricos y sus resultados (1 para victoria local, X para empate, 2 para victoria visitante).")

    # 1. Crear un dataset de entrenamiento de ejemplo
    # En un caso real, cargarías un CSV histórico aquí.
    # Simulamos datos históricos creando variaciones de los partidos actuales.
    historical_data = []
    for match in _match_data:
        for _ in range(20): # Crear 20 variaciones por partido
            home_elo_var = match['home_elo'] + np.random.randint(-50, 50)
            away_elo_var = match['away_elo'] + np.random.randint(-50, 50)
            elo_diff_var = home_elo_var - away_elo_var
            
            # Asignar un resultado basado en la diferencia de ELO (muy simplificado)
            if elo_diff_var > 100:
                result = 1 # Gana local
            elif elo_diff_var < -100:
                result = 2 # Gana visitante
            else:
                result = 0 # Empate

            historical_data.append({
                'elo_diff': elo_diff_var,
                'rating_diff': match['rating_diff'] + np.random.uniform(-5, 5),
                'home_goals_xg': match['home_goals_xg'] + np.random.uniform(-0.5, 0.5),
                'away_goals_xg': match['away_goals_xg'] + np.random.uniform(-0.5, 0.5),
                'result': result
            })
    
    df_train = pd.DataFrame(historical_data)

    # 2. Definir features (X) y target (y)
    features = ['elo_diff', 'rating_diff', 'home_goals_xg', 'away_goals_xg']
    X = df_train[features]
    y = df_train['result']

    # 3. Escalar los datos (buena práctica para muchos modelos)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Entrenar el modelo de Regresión Logística
    model = LogisticRegression(multi_class='ovr', solver='liblinear')
    model.fit(X_scaled, y)
    
    return model, scaler, features

# --- APLICACIÓN DE STREAMLIT ---

def main():
    st.set_page_config(page_title="Champions League Predictor", layout="wide")
    st.title("⚽ Sistema de Predicción Champions League")
    st.markdown("Usando Machine Learning para predecir los resultados de los partidos de hoy.")

    # Cargar y procesar datos
    teams_data, match_data = process_data()

    st.header("📊 Datos de los Equipos y Partidos")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estadísticas de Equipos")
        st.dataframe(pd.DataFrame(teams_data).T)
    
    with col2:
        st.subheader("Partidos a Predecir")
        st.dataframe(pd.DataFrame(match_data))

    st.header("🤖 Predicciones con Machine Learning")

    # Entrenar el modelo
    model, scaler, features = train_prediction_model(teams_data, match_data)
    st.success("✅ Modelo de Regresión Logística entrenado (con datos de demostración).")

    # Hacer predicciones para los partidos de hoy
    today_matches_df = pd.DataFrame(match_data)
    X_today = today_matches_df[features]
    X_today_scaled = scaler.transform(X_today)
    
    predictions = model.predict(X_today_scaled)
    prediction_probs = model.predict_proba(X_today_scaled)

    # Mostrar predicciones
    results_map = {0: 'Empate', 1: 'Victoria Local', 2: 'Victoria Visitante'}
    today_matches_df['Predicción ML'] = [results_map[p] for p in predictions]
    
    # Añadir probabilidades
    probs_df = pd.DataFrame(prediction_probs, columns=model.classes_)
    today_matches_df['Prob. Empate'] = probs_df[0].round(2) * 100
    today_matches_df['Prob. Victoria Local'] = probs_df[1].round(2) * 100
    today_matches_df['Prob. Victoria Visitante'] = probs_df[2].round(2) * 100

    st.subheader("Resultados Predichos")
    st.dataframe(today_matches_df[['home_team', 'away_team', 'Predicción ML', 'Prob. Victoria Local', 'Prob. Empate', 'Prob. Victoria Visitante']])


if __name__ == "__main__":
    main()
