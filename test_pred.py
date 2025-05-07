import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.preprocessing import StandardScaler

# Charger la liste des features utilisée lors de l'entraînement
with open('feature_cols.json', 'r') as f:
    feature_cols = json.load(f)

# Définir les dimensions d'entrée en fonction de la liste des features
seq_input_dim = len(feature_cols)
static_input_dim = len(feature_cols)

# Redéfinition du modèle (identique à celui utilisé lors de l'entraînement)
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
        x = self.seq_proj(seq_input)            # (batch_size, seq_length, d_model)
        x = x.transpose(0, 1)                     # (seq_length, batch_size, d_model)
        x = self.transformer_encoder(x)           # (seq_length, batch_size, d_model)
        x = x.mean(dim=0)                         # (batch_size, d_model)
        fusion = torch.cat([x, static_input], dim=1)  # (batch_size, d_model + static_input_dim)
        bp24_pred = self.bp24_regressor(fusion)
        bp3_pred = self.bp3_regressor(fusion)
        risk_logit = self.risk_classifier(fusion)
        return bp24_pred, bp3_pred, risk_logit

# Instanciation et chargement du modèle sauvegardé
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskBPModel(seq_input_dim, static_input_dim, d_model=64, nhead=8, num_layers=2, dropout=0.1)
model.load_state_dict(torch.load('model.pth', map_location=device))
model = model.to(device)
model.eval()

# Charger une instance d'entrée depuis le fichier des instances
df_instances = pd.read_csv('model_instances.csv')
example_row = df_instances.iloc[0]

# Convertir les colonnes JSON en tableaux numpy
seq_input = np.array(json.loads(example_row['seq_input']))      # forme: (seq_length, feature_dim)
static_input = np.array(json.loads(example_row['static_input']))  # forme: (feature_dim,)

# Préparer les tenseurs d'entrée (ajouter une dimension batch)
seq_input_tensor = torch.tensor(seq_input, dtype=torch.float32).unsqueeze(0).to(device)
static_input_tensor = torch.tensor(static_input, dtype=torch.float32).unsqueeze(0).to(device)

# Réaliser la prédiction
with torch.no_grad():
    bp24_pred, bp3_pred, risk_logit = model(seq_input_tensor, static_input_tensor)
    risk_prob = torch.sigmoid(risk_logit)

# Charger le scaler sauvegardé pour les cibles
# Ce scaler doit avoir été sauvegardé lors du prétraitement pour normaliser "ta_systolique_mean" et "ta_diastolique_mean"
with open('scaler_bp.pkl', 'rb') as f:
    scaler_bp = pickle.load(f)

# Inverser la normalisation pour BP24
bp24_pred_np = bp24_pred.cpu().numpy()  # Valeurs normalisées
bp24_real = scaler_bp.inverse_transform(bp24_pred_np)

# Inverser la normalisation pour BP24
bp3_pred_np = bp3_pred.cpu().numpy()  # Valeurs normalisées
bp3_real = scaler_bp.inverse_transform(bp3_pred_np)

#print("Prédiction BP24 (normalisées):", bp24_pred_np)
print("Prédiction BP24 (valeurs réelles):", bp24_real)
#print("Prédiction BP3:", bp3_pred.cpu().numpy())
print("Prédiction BP3 (valeurs réelles):", bp3_real)
#print("Logits de risque:", risk_logit.cpu().numpy())
print("Probabilité de risque:", risk_prob.cpu().numpy())
