import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Paramètres de simulation
n_individus = 1000
n_jours = 14
freq_minutes = 30              # Mesures toutes les 30 minutes
n_mesures_par_jour = int(24 * 60 / freq_minutes)  # 48 mesures par jour

# Liste pour stocker les DataFrames de chaque individu/jour
dfs = []

# Pour la reproductibilité
np.random.seed(42)

# Options de traitement et leur effet moyen sur la TA systolique (en mmHg)
# Ces valeurs s’appuient sur des données issues des guidelines JNC 7 (Chobanian et al., 2003) et ACC/AHA 2017 (Whelton et al., 2018)
treatment_effects = {
    "médicament": -10,
    "repos": -5,
    "activité sportive": -5,
    "eau": -2,
    "voir le médecin": -15,
    "médicament + repos": -12,
    "médicament + activité sportive": -12,
    "repos + eau": -6,
    "médicament + repos + activité sportive": -15,
    "voir le médecin + médicament": -18
}
treatment_options = list(treatment_effects.keys())

# Fonction pour catégoriser l'évolution de la TA en fonction du delta (en mmHg)
def categorize_bp_evolution(delta):
    if delta <= -8:
        return "amélioration significative"
    elif -8 < delta <= -3:
        return "amélioration modérée"
    elif -3 < delta < 3:
        return "stabilité"
    else:
        return "détérioration"

# Boucle sur les individus
for individu in range(n_individus):
    
    # Informations démographiques pour l'individu
    age = np.random.randint(30, 81)  # entre 30 et 80 ans
    sexe = np.random.choice(['M', 'F'])
    antecedents = np.random.choice(["hypertension", "aucun"], p=[0.3, 0.7])
    poids = np.random.normal(75, 10)       # en kg
    taille = np.random.normal(175, 7)        # en cm
    IMC = round(poids / ((taille/100)**2), 1)
    
    # Risque de complication de base en fonction des antécédents
    p_base = 0.20 if antecedents == "hypertension" else 0.05
    
    # Boucle sur les jours de suivi
    for j in range(n_jours):
        jour_date = datetime(2025, 3, 19) + timedelta(days=j)
        
        # Génération des timestamps pour ce jour (48 mesures)
        timestamps = [jour_date + timedelta(minutes=freq_minutes * i) for i in range(n_mesures_par_jour)]
        day_df = pd.DataFrame({'timestamp': timestamps})
        day_df['day_of_week'] = day_df['timestamp'].dt.day_name()
        
        # Vecteur temporel pour simuler le rythme circadien
        t = np.linspace(0, 2*np.pi, n_mesures_par_jour)
        
        # Simulation des mesures de TA (inspirées du rythme circadien)
        day_df['ta_systolique'] = 120 + 10 * np.sin(t) + np.random.normal(0, 3, n_mesures_par_jour)
        day_df['ta_diastolique'] = 80 + 5 * np.sin(t - np.pi/6) + np.random.normal(0, 2, n_mesures_par_jour)
        
        # Fréquence cardiaque et HRV
        day_df['heart_rate'] = 70 + 5 * np.sin(t - np.pi/3) + np.random.normal(0, 4, n_mesures_par_jour)
        day_df['hrv'] = 50 + np.random.normal(0, 5, n_mesures_par_jour)
        
        # Simulation de l'activité physique : pas et type d'activité en fonction de l'heure
        steps = []
        activity_types = []
        for ts in day_df['timestamp']:
            hour = ts.hour
            if 7 <= hour < 10:
                s = np.random.poisson(200)
                act = np.random.choice(['marche', 'repos'])
            elif 10 <= hour < 12:
                s = np.random.poisson(150)
                act = 'repos'
            elif 12 <= hour < 14:
                s = np.random.poisson(100)
                act = 'repos'
            elif 14 <= hour < 18:
                s = np.random.poisson(180)
                act = np.random.choice(['marche', 'exercice'])
            elif 18 <= hour < 21:
                s = np.random.poisson(220)
                act = np.random.choice(['exercice', 'marche'])
            else:
                s = np.random.poisson(20)
                act = 'repos'
            steps.append(s)
            activity_types.append(act)
        day_df['steps'] = steps
        day_df['activity_type'] = activity_types
        
        # Simulation du niveau de stress (échelle 1 à 10)
        day_df['stress_level'] = 5 + 2 * np.sin(t - np.pi/4) + np.random.normal(0, 1, n_mesures_par_jour)
        day_df['stress_level'] = day_df['stress_level'].round().clip(1, 10)
        
        # Statut de sommeil en fonction de l'heure (dormir entre 23h et 7h)
        day_df['sleep_status'] = day_df['timestamp'].dt.hour.apply(lambda h: 'dormir' if (h >= 23 or h < 7) else 'éveil')
        day_df['position'] = day_df['sleep_status'].apply(lambda x: 'allongé' if x == 'dormir' else 'assis')
        
        # Médication : prise à 8h et 20h
        day_df['medication_taken'] = day_df['timestamp'].apply(lambda ts: 1 if ts.hour in [8, 20] else 0)
        day_df['medication_info'] = day_df['medication_taken'].apply(lambda x: "MedA 10mg" if x==1 else "aucun")
        
        # Données environnementales (simulées autour de Paris)
        day_df['latitude'] = 48.8566 + np.random.normal(0, 0.005, n_mesures_par_jour)
        day_df['longitude'] = 2.3522 + np.random.normal(0, 0.005, n_mesures_par_jour)
        day_df['temperature'] = 12 + 8 * np.sin(t - np.pi/3) + np.random.normal(0, 1, n_mesures_par_jour)
        day_df['humidity'] = 70 + 10 * np.cos(t) + np.random.normal(0, 3, n_mesures_par_jour)
        
        # Indicateur hypertensif par mesure
        day_df['hypertension_event'] = ((day_df['ta_systolique'] > 140) | (day_df['ta_diastolique'] > 90)).astype(int)
        
        # Notes/commentaires (20% de chance)
        day_df['notes'] = [np.random.choice(["après repas", "stressé", "avant médicament", "en déplacement"]) 
                           if np.random.rand() < 0.2 else "" for _ in range(n_mesures_par_jour)]
        
        # Ajout des informations démographiques pour l'individu
        day_df['id_individu'] = individu
        day_df['age'] = age
        day_df['sexe'] = sexe
        day_df['antécédents'] = antecedents
        day_df['poids'] = poids
        day_df['taille'] = taille
        day_df['IMC'] = IMC
        
        # Simulation journalière d'un événement complication en fonction de la moyenne journalière de TA et de stress
        mean_systolic = day_df['ta_systolique'].mean()
        mean_stress = day_df['stress_level'].mean()
        additional_risk = 0.0
        if mean_systolic > 140:
            additional_risk += (mean_systolic - 140) / 100  # effet linéaire
        additional_risk += (mean_stress - 5) * 0.01
        p_day = np.clip(p_base + additional_risk, 0, 1)
        complication_event = np.random.choice([1, 0], p=[p_day, 1-p_day])
        day_df['complication_event'] = complication_event
        
        # Si un événement complication est détecté, simuler le traitement et l'évolution de la TA sur les 3 jours suivants
        if complication_event == 1:
            treatment_type = np.random.choice(treatment_options)
            # L'effet moyen du traitement sur la TA systolique (en mmHg)
            mean_delta = treatment_effects[treatment_type]
            # Variabilité individuelle
            bp_delta = mean_delta + np.random.normal(0, 2)
            bp_evolution = categorize_bp_evolution(bp_delta)
            # Consultation : si "voir le médecin" est inclus dans le traitement, la consultation est assurée
            if "voir le médecin" in treatment_type:
                visited_doctor = 1
            else:
                visited_doctor = np.random.choice([1, 0], p=[0.3, 0.7])
        else:
            treatment_type = "aucun"
            bp_delta = 0
            bp_evolution = "aucune"
            visited_doctor = 0
        
        # Ajout des colonnes de traitement et de suivi pour le jour (répété sur toutes les mesures)
        day_df['treatment_type'] = treatment_type
        day_df['bp_delta_next3days'] = round(bp_delta, 1)
        day_df['bp_evolution_next3days'] = bp_evolution
        day_df['visited_doctor'] = visited_doctor
        
        # Réorganisation des colonnes
        colonnes = ['id_individu', 'timestamp', 'day_of_week', 'age', 'sexe', 'antécédents',
                    'ta_systolique', 'ta_diastolique', 'heart_rate', 'hrv',
                    'steps', 'activity_type', 'stress_level', 'sleep_status', 'position',
                    'medication_taken', 'medication_info',
                    'poids', 'taille', 'IMC',
                    'latitude', 'longitude', 'temperature', 'humidity',
                    'hypertension_event', 'complication_event',
                    'treatment_type', 'bp_delta_next3days', 'bp_evolution_next3days', 'visited_doctor',
                    'notes']
        day_df = day_df[colonnes]
        
        dfs.append(day_df)

# Concaténer toutes les données
df_final = pd.concat(dfs, ignore_index=True)
# Enregistrer
df_final.to_csv('simulated_bp_data.csv', index=False)

print("Nombre total de lignes:", df_final.shape[0])
print(df_final.head(10))
