import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb

# Configuración de la página
st.set_page_config(
    page_title="🏆 Champions League ML Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🏆 Champions League ML Dashboard")
st.markdown("### 📅 Partidos de Hoy - Miércoles 5/11/2025")

# Sidebar
st.sidebar.header("🎯 Configuración")
st.sidebar.markdown("---")

@st.cache_data
def load_data():
    """Cargar datos de equipos y partidos"""
    try:
        with open('data/teams_data.json', 'r') as f:
            teams_data = json.load(f)
        
        with open('data/matches_data.json', 'r') as f:
            matches_data = json.load(f)
        
        return teams_data, matches_data
    except:
        # Datos de respaldo si no se encuentran archivos
        teams_data = {
            "Qarabağ": {"rating": 78, "tilt": 0.213, "elo_general": 769, "elo_country": 1, "prob_victoria": 0.186, "goles_esperados": 2.74, "posicion_actual": 15, "puntos_actuales": 6},
            "Chelsea": {"rating": 91, "tilt": 0.161, "elo_general": 38, "elo_country": 4, "prob_victoria": 0.607, "goles_esperados": 2.0, "posicion_actual": 13, "puntos_actuales": 6},
            "Inter": {"rating": 94, "tilt": 0.08, "elo_general": 19, "elo_country": 2, "prob_victoria": 0.84, "goles_esperados": 2.74, "posicion_actual": 4, "puntos_actuales": 9},
            "Kairat Almaty": {"rating": 80, "tilt": -0.012, "elo_general": 476, "elo_country": 1, "prob_victoria": 0.046, "goles_esperados": 0.5, "posicion_actual": 34, "puntos_actuales": 1},
            "Man. City": {"rating": 93, "tilt": 0.185, "elo_general": 22, "elo_country": 3, "prob_victoria": 0.578, "goles_esperados": 2.24, "posicion_actual": 9, "puntos_actuales": 7},
            "B. Dortmund": {"rating": 90, "tilt": 0.085, "elo_general": 45, "elo_country": 4, "prob_victoria": 0.223, "goles_esperados": 1.34, "posicion_actual": 8, "puntos_actuales": 7},
            "Club Brugge": {"rating": 90, "tilt": 0.026, "elo_general": 54, "elo_country": 2, "prob_victoria": 0.194, "goles_esperados": 2.0, "posicion_actual": 24, "puntos_actuales": 3},
            "FC Barcelona": {"rating": 95, "tilt": 0.015, "elo_general": 13, "elo_country": 3, "prob_victoria": 0.61, "goles_esperados": 2.25, "posicion_actual": 12, "puntos_actuales": 6}
        }
        
        matches_data = [
            {"home_team": "Qarabağ", "away_team": "Chelsea", "home_win_prob": 0.186, "away_win_prob": 0.607, "draw_prob": 0.207, "elo_diff": 731, "rating_diff": -13},
            {"home_team": "Inter", "away_team": "Kairat Almaty", "home_win_prob": 0.84, "away_win_prob": 0.046, "draw_prob": 0.114, "elo_diff": -457, "rating_diff": 14},
            {"home_team": "Man. City", "away_team": "B. Dortmund", "home_win_prob": 0.578, "away_win_prob": 0.223, "draw_prob": 0.199, "elo_diff": -23, "rating_diff": 3},
            {"home_team": "Club Brugge", "away_team": "FC Barcelona", "home_win_prob": 0.194, "away_win_prob": 0.61, "draw_prob": 0.196, "elo_diff": 41, "rating_diff": -5}
        ]
        
        return teams_data, matches_data

@st.cache_resource
def train_model():
    """Entrenar el modelo ML"""
    np.random.seed(42)
    n_historical = 1000
    
    historical_data = []
    for i in range(n_historical):
        home_rating = np.random.uniform(70, 100)
        away_rating = np.random.uniform(70, 100)
        home_elo = np.random.randint(1, 500)
        away_elo = np.random.randint(1, 500)
        
        rating_diff = home_rating - away_rating
        elo_diff = home_elo - away_elo
        
        win_prob = 1 / (1 + np.exp(-(rating_diff + elo_diff/50)/20))
        draw_prob = 0.25
        
        rand = np.random.random()
        if rand < win_prob:
            result = 'Home Win'
        elif rand < win_prob + draw_prob:
            result = 'Draw'
        else:
            result = 'Away Win'
        
        historical_data.append({
            'home_rating': home_rating,
            'away_rating': away_rating,
            'home_elo': home_elo,
            'away_elo': away_elo,
            'rating_diff': rating_diff,
            'elo_diff': elo_diff,
            'result': result
        })
    
    historical_df = pd.DataFrame(historical_data)
    
    feature_columns = ['home_rating', 'away_rating', 'home_elo', 'away_elo', 'rating_diff', 'elo_diff']
    X = historical_df[feature_columns]
    y = historical_df['result']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train_scaled, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(scaler.transform(X_test)))
    
    return model, scaler, le, accuracy

def main():
    # Cargar datos
    teams_data, matches_data = load_data()
    teams_df = pd.DataFrame.from_dict(teams_data, orient='index').reset_index()
    teams_df.rename(columns={'index': 'team'}, inplace=True)
    
    # Entrenar modelo
    model, scaler, le, model_accuracy = train_model()
    
    # Preparar predicciones
    today_matches_features = []
    for match in matches_data:
        home_team = match['home_team']
        away_team = match['away_team']
        
        home_stats = teams_df[teams_df['team'] == home_team].iloc[0]
        away_stats = teams_df[teams_df['team'] == away_team].iloc[0]
        
        features = [
            home_stats['rating'],
            away_stats['rating'],
            home_stats['elo_general'],
            away_stats['elo_general'],
            home_stats['rating'] - away_stats['rating'],
            home_stats['elo_general'] - away_stats['elo_general']
        ]
        today_matches_features.append(features)
    
    today_features_scaled = scaler.transform(np.array(today_matches_features))
    predictions_encoded = model.predict(today_features_scaled)
    prediction_proba = model.predict_proba(today_features_scaled)
    predictions_labels = le.inverse_transform(predictions_encoded)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏟️ Partidos", "🤖 ML Predictions", "📊 Análisis", "📈 Insights"])
    
    with tab1:
        st.header("🏟️ Partidos de Hoy")
        
        col1, col2 = st.columns(2)
        
        for i, match in enumerate(matches_data):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"### {match['home_team']} vs {match['away_team']}")
                
                # Información básica
                st.markdown(f"""
                **📊 Estadísticas:**
                - Rating: {match.get('home_rating', 'N/A')} vs {match.get('away_rating', 'N/A')}
                - ELO: {match.get('home_elo', 'N/A')} vs {match.get('away_elo', 'N/A')}
                - xG: {match.get('home_goals_xg', 'N/A')} vs {match.get('away_goals_xg', 'N/A')}
                """)
                
                # Probabilidades
                prob_data = {
                    'Resultado': [match['home_team'], 'Empate', match['away_team']],
                    'Probabilidad': [match['home_win_prob']*100, match['draw_prob']*100, match['away_win_prob']*100]
                }
                prob_df = pd.DataFrame(prob_data)
                
                fig = px.bar(prob_df, x='Resultado', y='Probabilidad', 
                           color='Probabilidad', color_continuous_scale='Blues')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
    
    with tab2:
        st.header("🤖 Predicciones Machine Learning")
        st.markdown(f"**🎯 Precisión del modelo: {model_accuracy:.1%}**")
        
        predictions_results = []
        for i, match in enumerate(matches_data):
            home_team = match['home_team']
            away_team = match['away_team']
            ml_prediction = predictions_labels[i]
            
            prob_home = prediction_proba[i][le.transform(['Home Win'])[0]]
            prob_draw = prediction_proba[i][le.transform(['Draw'])[0]]
            prob_away = prediction_proba[i][le.transform(['Away Win'])[0]]
            
            result = {
                'match': f"{home_team} vs {away_team}",
                'ml_prediction': ml_prediction,
                'ml_probabilities': {'home': prob_home, 'draw': prob_draw, 'away': prob_away},
                'original_probabilities': {'home': match['home_win_prob'], 'draw': match['draw_prob'], 'away': match['away_win_prob']}
            }
            predictions_results.append(result)
        
        for result in predictions_results:
            st.markdown(f"### {result['match']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🤖 Predicción ML**")
                ml_data = {
                    'Resultado': [result['match'].split(' vs ')[0], 'Empate', result['match'].split(' vs ')[1]],
                    'Probabilidad': [result['ml_probabilities']['home']*100, 
                                   result['ml_probabilities']['draw']*100, 
                                   result['ml_probabilities']['away']*100]
                }
                ml_df = pd.DataFrame(ml_data)
                fig1 = px.bar(ml_df, x='Resultado', y='Probabilidad', 
                             color='Probabilidad', color_continuous_scale='Reds')
                fig1.update_layout(height=300, title="Modelo ML")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.markdown("**📊 Probabilidades Originales**")
                orig_data = {
                    'Resultado': [result['match'].split(' vs ')[0], 'Empate', result['match'].split(' vs ')[1]],
                    'Probabilidad': [result['original_probabilities']['home']*100, 
                                   result['original_probabilities']['draw']*100, 
                                   result['original_probabilities']['away']*100]
                }
                orig_df = pd.DataFrame(orig_data)
                fig2 = px.bar(orig_df, x='Resultado', y='Probabilidad', 
                             color='Probabilidad', color_continuous_scale='Blues')
                fig2.update_layout(height=300, title="Probabilidades Originales")
                st.plotly_chart(fig2, use_container_width=True)
            
            # Mostrar recomendación
            max_prob = max(result['ml_probabilities'].values())
            if max_prob == result['ml_probabilities']['home']:
                recommendation = f"🏠 Victoria {result['match'].split(' vs ')[0]}"
            elif max_prob == result['ml_probabilities']['away']:
                recommendation = f"✈️ Victoria {result['match'].split(' vs ')[1]}"
            else:
                recommendation = "🤝 Empate"
            
            confidence_level = "Alta 🔥" if max_prob > 0.6 else "Media ⚡" if max_prob > 0.4 else "Baja ⚠️"
            
            st.success(f"**🎯 Recomendación:** {recommendation} (Confianza: {confidence_level} - {max_prob:.1%})")
            st.markdown("---")
    
    with tab3:
        st.header("📊 Análisis de Equipos")
        
        # Filtro de equipos
        selected_teams = st.multiselect("Seleccionar equipos:", teams_df['team'].tolist())
        if not selected_teams:
            selected_teams = teams_df['team'].tolist()
        
        filtered_teams = teams_df[teams_df['team'].isin(selected_teams)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Ratings de Equipos")
            fig = px.bar(filtered_teams, x='team', y='rating', 
                        color='rating', color_continuous_scale='Viridis')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Goles Esperados (xG)")
            fig = px.bar(filtered_teams, x='team', y='goles_esperados',
                        color='goles_esperados', color_continuous_scale='Plasma')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### ⚡ Tilt (Forma)")
            fig = px.bar(filtered_teams, x='team', y='tilt',
                        color='tilt', color_continuous_scale='RdBu')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.markdown("### 🏅 Puntos en Champions")
            fig = px.bar(filtered_teams, x='team', y='puntos_actuales',
                        color='puntos_actuales', color_continuous_scale='Greens')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de estadísticas
        st.markdown("### 📋 Tabla de Estadísticas Detalladas")
        st.dataframe(filtered_teams, use_container_width=True)
    
    with tab4:
        st.header("📈 Insights y Recomendaciones")
        
        st.markdown("### 🎯 Resumen de Predicciones")
        
        # Análisis de confianza
        high_confidence = []
        medium_confidence = []
        low_confidence = []
        
        for result in predictions_results:
            confidence = max(result['ml_probabilities'].values())
            if confidence > 0.6:
                high_confidence.append(result['match'])
            elif confidence > 0.4:
                medium_confidence.append(result['match'])
            else:
                low_confidence.append(result['match'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔥 Alta Confianza", len(high_confidence))
            for match in high_confidence:
                st.success(f"• {match}")
        
        with col2:
            st.metric("⚡ Media Confianza", len(medium_confidence))
            for match in medium_confidence:
                st.warning(f"• {match}")
        
        with col3:
            st.metric("⚠️ Baja Confianza", len(low_confidence))
            for match in low_confidence:
                st.error(f"• {match}")
        
        st.markdown("### 💡 Recomendaciones Estratégicas")
        st.markdown("""
        **🎯 Estrategias de Apuesta:**
        - **Partidos Alta Confianza:** Oportunidades principales para apostar
        - **Partidos Media Confianza:** Combinar con otras fuentes de análisis
        - **Partidos Baja Confianza:** Evitar o apostar cantidades mínimas
        
        **📊 Consideraciones del Modelo:**
        - Precisión del modelo basada en datos históricos simulados
        - Considera ratings, ELO, tilt y estadísticas actuales
        - Actualizaciones recomendadas después de cada jornada
        """)
        
        st.markdown("### ⚠️ Disclaimer")
        st.warning("""
        **Importante:**
        - Este análisis es solo para fines educativos
        - Las apuestas deportivas implican riesgo financiero
        - Juega responsablemente
        - No garantizamos resultados
        """)

if __name__ == "__main__":
    main()
