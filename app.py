import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# --- FUNCIONES DE PROCESAMIENTO DE DATOS (Tus funciones, ligeramente adaptadas) ---

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
def process_excel_data(uploaded_file):
    """Procesa datos desde archivo Excel subido por el usuario"""
    try:
        df = pd.read_excel(uploaded_file, header=None)
        
        # Nombres de equipos
        team_names = ['Qarabağ', 'Chelsea', 'Inter', 'Kairat Almaty', 'Man. City', 'B. Dortmund', 'Club Brugge', 'FC Barcelona']
        
        # Extraer datos
        ratings = df.iloc[2, 2:10].values
        tilts = df.iloc[3, 2:10].values
        elo_general = df.iloc[4, 2:10].values
        elo_country = df.iloc[5, 2:10].values
        prob_victoria = df.iloc[6, 2:10].values
        prob_empate = df.iloc[7, 2:10].values
        goles_esperados = df.iloc[8, 2:10].values
        posicion_actual = df.iloc[9, 2:10].values
        puntos_actuales = df.iloc[10, 2:10].values
        
        teams_data = {}
        for i, team in enumerate(team_names):
            teams_data[team] = {
                'rating': clean_numeric(ratings[i]),
                'tilt': clean_percentage(tilts[i]),
                'elo_general': int(clean_numeric(elo_general[i])),
                'elo_country': int(clean_numeric(elo_country[i])),
                'prob_victoria': clean_percentage(prob_victoria[i]),
