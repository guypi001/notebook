import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
import pickle

# -------------------------
# Prétraitement et Feature Engineering
# -------------------------

# Chargement du dataset simulé
df = pd.read_csv('simulated_bp_data.csv')

# Conversion de la colonne 'timestamp' en datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Création d'une colonne "date" pour l'agrégation journalière
df['date'] = df['timestamp'].dt.date

# Définition des fonctions d'agrégation pour les variables numériques, en ajoutant complication_event
agg_funcs = {
    'ta_systolique': ['mean', 'std', 'min', 'max'],
    'ta_diastolique': ['mean', 'std', 'min', 'max'],
    'heart_rate': ['mean', 'std'],
    'stress_level': ['mean'],
    'steps': ['sum'],  # Total de pas dans la journée
    'temperature': ['mean', 'std'],
    'humidity': ['mean', 'std'],
    'complication_event': 'max'  # Conserver le flag de complication pour la journée
}

df_daily = df.groupby(['id_individu', 'date']).agg(agg_funcs)
# Aplatir les colonnes multi-index
df_daily.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df_daily.columns]
df_daily = df_daily.reset_index()

# Calcul de la variation journalière (delta = max - min) pour la TA
df_daily['ta_systolic_delta'] = df_daily['ta_systolique_max'] - df_daily['ta_systolique_min']
df_daily['ta_diastolic_delta'] = df_daily['ta_diastolique_max'] - df_daily['ta_diastolique_min']

# Fusion des variables statiques (démographiques) par individu
info_cols = ['id_individu', 'age', 'sexe', 'antécédents', 'poids', 'taille', 'IMC']
df_info = df.drop_duplicates(subset=['id_individu'])[info_cols]
df_daily = pd.merge(df_daily, df_info, on='id_individu', how='left')

# Prétraitement des variables catégorielles issues des mesures horaires
def mode_series(series):
    return series.mode().iloc[0] if not series.mode().empty else series.iloc[0]

cat_cols = ['day_of_week', 'activity_type', 'sleep_status', 'position', 'medication_info', 
            'treatment_type', 'bp_evolution_next3days']
df_cat = df.groupby(['id_individu', 'date'])[cat_cols].agg(lambda x: mode_series(x)).reset_index()

# Fusion des features catégorielles avec les données agrégées journalières
df_daily = pd.merge(df_daily, df_cat, on=['id_individu', 'date'], how='left')

# Application du one-hot encoding sur les variables catégorielles
df_encoded = pd.get_dummies(df_daily, columns=cat_cols, drop_first=True)

# Vérification et conversion forcée en numérique pour toutes les colonnes de type "object"
print("Types avant conversion:")
print(df_encoded.dtypes)
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
print("Types après conversion:")
print(df_encoded.dtypes)

# Remplir les valeurs manquantes au lieu de dropna
df_encoded = df_encoded.fillna(0)
if df_encoded.empty:
    raise ValueError("Le DataFrame prétraité est vide après fillna. Vérifiez vos étapes de prétraitement.")

# Normalisation des variables numériques
num_cols = ['ta_systolique_mean', 'ta_systolique_std', 'ta_systolique_min', 'ta_systolique_max',
            'ta_diastolique_mean', 'ta_diastolique_std', 'ta_diastolique_min', 'ta_diastolique_max',
            'heart_rate_mean', 'heart_rate_std', 'stress_level_mean', 'steps_sum',
            'ta_systolic_delta', 'ta_diastolic_delta', 'age', 'poids', 'taille', 'IMC',
            'temperature_mean', 'temperature_std', 'humidity_mean', 'humidity_std']

scaler = StandardScaler()
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# Enregistrement du dataset prétraité
df_encoded.to_csv('preprocessed_simulated_bp_data.csv', index=False)
print("Nombre de lignes (agrégées journalières) :", df_encoded.shape[0])
print(df_encoded.head(10))

# -------------------------
# Sauvegarde du scaler pour les cibles BP24 (ta_systolique_mean et ta_diastolique_mean)
# -------------------------
target_cols = ['ta_systolique_mean', 'ta_diastolique_mean']
# Ici, on utilise les valeurs avant normalisation qui se trouvent dans df_daily pour ajuster le scaler
scaler_bp = StandardScaler()
scaler_bp.fit(df_daily[target_cols])
with open('scaler_bp.pkl', 'wb') as f:
    pickle.dump(scaler_bp, f)
print("Scaler pour BP24 sauvegardé dans 'scaler_bp.pkl'.")

# -------------------------
# Création des cibles et du dataset pour le modèle séquentiel
# -------------------------
df_encoded = df_encoded.sort_values(['id_individu', 'date'])
df_encoded['bp24_target_systolic'] = df_encoded.groupby('id_individu')['ta_systolique_mean'].shift(-1)
df_encoded['bp24_target_diastolic'] = df_encoded.groupby('id_individu')['ta_diastolique_mean'].shift(-1)
df_encoded['bp3_target_systolic'] = df_encoded.groupby('id_individu')['ta_systolique_mean'].shift(-3) - df_encoded['ta_systolique_mean']
df_encoded['bp3_target_diastolic'] = df_encoded.groupby('id_individu')['ta_diastolique_mean'].shift(-3) - df_encoded['ta_diastolique_mean']
df_encoded['risk_target'] = df_encoded.groupby('id_individu')['complication_event_max'].shift(-1)

df_model = df_encoded.dropna(subset=['bp24_target_systolic', 'bp24_target_diastolic', 
                                       'bp3_target_systolic', 'bp3_target_diastolic', 'risk_target']).copy()

exclude_cols = ['id_individu', 'date', 'bp24_target_systolic', 'bp24_target_diastolic',
                'bp3_target_systolic', 'bp3_target_diastolic', 'risk_target']
feature_cols = [col for col in df_model.columns if col not in exclude_cols]

# Sauvegarde de la liste des features pour utilisation lors des prédictions
with open('feature_cols.json', 'w') as f:
    json.dump(feature_cols, f)
print("Liste des features enregistrée dans 'feature_cols.json'.")

data_instances = []
for id_indiv, group in df_model.groupby('id_individu'):
    group = group.sort_values('date').reset_index(drop=True)
    if len(group) < 4:
        continue
    for i in range(3, len(group)):
        seq_input = group.loc[i-3:i-1, feature_cols].values
        static_input = group.loc[i, feature_cols].values
        target_bp24 = group.loc[i, ['bp24_target_systolic', 'bp24_target_diastolic']].values.astype(np.float32)
        target_bp3 = group.loc[i, ['bp3_target_systolic', 'bp3_target_diastolic']].values.astype(np.float32)
        target_risk = np.array([group.loc[i, 'risk_target']], dtype=np.float32)
        data_instances.append({
            'id_individu': id_indiv,
            'seq_input': seq_input,
            'static_input': static_input,
            'target_bp24': target_bp24,
            'target_bp3': target_bp3,
            'target_risk': target_risk
        })

print(f"Nombre d'instances : {len(data_instances)}")

unique_ids = df_model['id_individu'].unique()
train_ids, test_ids = train_test_split(unique_ids, test_size=0.15, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.15/0.85, random_state=42)

train_instances = [inst for inst in data_instances if inst['id_individu'] in train_ids]
val_instances = [inst for inst in data_instances if inst['id_individu'] in val_ids]
test_instances = [inst for inst in data_instances if inst['id_individu'] in test_ids]

print(f"Instances Train: {len(train_instances)}, Val: {len(val_instances)}, Test: {len(test_instances)}")

# -------------------------
# Enregistrement du dataset des instances au format CSV
# -------------------------
rows = []
for inst in data_instances:
    row = {
        'id_individu': inst['id_individu'],
        'seq_input': json.dumps(inst['seq_input'].tolist()),
        'static_input': json.dumps(inst['static_input'].tolist()),
        'target_bp24': json.dumps(inst['target_bp24'].tolist()),
        'target_bp3': json.dumps(inst['target_bp3'].tolist()),
        'target_risk': json.dumps(inst['target_risk'].tolist())
    }
    rows.append(row)
df_instances = pd.DataFrame(rows)
df_instances.to_csv('model_instances.csv', index=False)
print("Le dataset des instances a été enregistré sous 'model_instances.csv'.")

# -------------------------
# Définition du Dataset PyTorch avec conversion forcée en float32
# -------------------------
class BPTimeSeriesDataset(Dataset):
    def __init__(self, instances):
        self.instances = instances
    def __len__(self):
        return len(self.instances)
    def __getitem__(self, idx):
        inst = self.instances[idx]
        seq_input = torch.tensor(np.array(inst['seq_input'], dtype=np.float32))
        static_input = torch.tensor(np.array(inst['static_input'], dtype=np.float32))
        target_bp24 = torch.tensor(np.array(inst['target_bp24'], dtype=np.float32))
        target_bp3 = torch.tensor(np.array(inst['target_bp3'], dtype=np.float32))
        target_risk = torch.tensor(np.array(inst['target_risk'], dtype=np.float32))
        return seq_input, static_input, target_bp24, target_bp3, target_risk

train_dataset = BPTimeSeriesDataset(train_instances)
val_dataset = BPTimeSeriesDataset(val_instances)
test_dataset = BPTimeSeriesDataset(test_instances)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------
# Définition du modèle MultiTaskBPModel
# -------------------------
seq_input_dim = len(feature_cols)
static_input_dim = len(feature_cols)

class MultiTaskBPModel(nn.Module):
    def __init__(self, seq_input_dim, static_input_dim, d_model=64, nhead=8, num_layers=2, dropout=0.1):
        super(MultiTaskBPModel, self).__init__()
        self.seq_proj = nn.Linear(seq_input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        fusion_dim = d_model + static_input_dim
        self.bp24_regressor = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.bp3_regressor = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.risk_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, seq_input, static_input):
        x = self.seq_proj(seq_input)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        fusion = torch.cat([x, static_input], dim=1)
        bp24_pred = self.bp24_regressor(fusion)
        bp3_pred = self.bp3_regressor(fusion)
        risk_logit = self.risk_classifier(fusion)
        return bp24_pred, bp3_pred, risk_logit

model = MultiTaskBPModel(seq_input_dim, static_input_dim, d_model=64, nhead=8, num_layers=2, dropout=0.1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

criterion_reg = nn.MSELoss()
criterion_bce = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for seq_input, static_input, target_bp24, target_bp3, target_risk in train_loader:
        seq_input = seq_input.to(device)
        static_input = static_input.to(device)
        target_bp24 = target_bp24.to(device)
        target_bp3 = target_bp3.to(device)
        target_risk = target_risk.to(device)
        
        optimizer.zero_grad()
        pred_bp24, pred_bp3, risk_logit = model(seq_input, static_input)
        loss_bp24 = criterion_reg(pred_bp24, target_bp24)
        loss_bp3 = criterion_reg(pred_bp3, target_bp3)
        loss_risk = criterion_bce(risk_logit, target_risk)
        loss = loss_bp24 + loss_bp3 + loss_risk
        if torch.isnan(loss):
            print("Loss is NaN, skipping batch")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())
    
    train_loss = np.mean(train_losses) if train_losses else float('nan')
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for seq_input, static_input, target_bp24, target_bp3, target_risk in val_loader:
            seq_input = seq_input.to(device)
            static_input = static_input.to(device)
            target_bp24 = target_bp24.to(device)
            target_bp3 = target_bp3.to(device)
            target_risk = target_risk.to(device)
            
            pred_bp24, pred_bp3, risk_logit = model(seq_input, static_input)
            loss_bp24 = criterion_reg(pred_bp24, target_bp24)
            loss_bp3 = criterion_reg(pred_bp3, target_bp3)
            loss_risk = criterion_bce(risk_logit, target_risk)
            loss = loss_bp24 + loss_bp3 + loss_risk
            val_losses.append(loss.item())
    
    val_loss = np.mean(val_losses) if val_losses else float('nan')
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

model.eval()
test_losses = []
with torch.no_grad():
    for seq_input, static_input, target_bp24, target_bp3, target_risk in test_loader:
        seq_input = seq_input.to(device)
        static_input = static_input.to(device)
        target_bp24 = target_bp24.to(device)
        target_bp3 = target_bp3.to(device)
        target_risk = target_risk.to(device)
        
        pred_bp24, pred_bp3, risk_logit = model(seq_input, static_input)
        loss_bp24 = criterion_reg(pred_bp24, target_bp24)
        loss_bp3 = criterion_reg(pred_bp3, target_bp3)
        loss_risk = criterion_bce(risk_logit, target_risk)
        loss = loss_bp24 + loss_bp3 + loss_risk
        test_losses.append(loss.item())
test_loss = np.mean(test_losses) if test_losses else float('nan')
print(f"Test Loss: {test_loss:.6f}")

torch.save(model.state_dict(), 'model.pth')
print("Modèle sauvegardé dans 'model.pth'.")
