import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Chargement du dataset prétraité
df = pd.read_csv('preprocessed_simulated_bp_data.csv')
df['date'] = pd.to_datetime(df['date'])

# 1.a Création des cibles en décalant les moyennes journalières
# Pour chaque individu, nous définissons :
# - bp24_target : la TA moyenne du jour suivant (pour les 2 composantes systolique et diastolique)
# - bp3_target : la différence entre le jour d+3 et le jour courant
# - risk_target : le flag complication du jour suivant
df = df.sort_values(['id_individu', 'date'])
df['bp24_target_systolic'] = df.groupby('id_individu')['ta_systolique_mean'].shift(-1)
df['bp24_target_diastolic'] = df.groupby('id_individu')['ta_diastolique_mean'].shift(-1)
df['bp3_target_systolic'] = df.groupby('id_individu')['ta_systolique_mean'].shift(-3) - df['ta_systolique_mean']
df['bp3_target_diastolic'] = df.groupby('id_individu')['ta_diastolique_mean'].shift(-3) - df['ta_diastolique_mean']
df['risk_target'] = df.groupby('id_individu')['complication_event_max'].shift(-1)
df_model = df.dropna(subset=['bp24_target_systolic', 'bp24_target_diastolic', 'bp3_target_systolic', 'bp3_target_diastolic', 'risk_target']).copy()

# 2. Construction d'un dataset pour le modèle séquentiel
# On va créer des instances avec une séquence d'historique de 3 jours pour chaque individu.
# On utilisera l'ensemble des features issues de la prétraitement, en excluant les colonnes d'ID, date et cibles.

exclude_cols = ['id_individu', 'date', 'bp24_target_systolic', 'bp24_target_diastolic',
                'bp3_target_systolic', 'bp3_target_diastolic', 'risk_target']
feature_cols = [col for col in df_model.columns if col not in exclude_cols]

# Pour construire les séquences, on groupe par individu
data_instances = []
for id_indiv, group in df_model.groupby('id_individu'):
    group = group.sort_values('date').reset_index(drop=True)
    # On ne peut créer une instance que si on dispose d'au moins 4 jours consécutifs
    if len(group) < 4:
        continue
    # Pour i allant de 3 à len(group)-1 (car la cible est déjà décalée pour le jour i)
    for i in range(3, len(group)):
        seq_input = group.loc[i-3:i-1, feature_cols].values  # les 3 jours précédents
        static_input = group.loc[i, feature_cols].values      # le jour courant
        target_bp24 = group.loc[i, ['bp24_target_systolic', 'bp24_target_diastolic']].values.astype(np.float32)
        target_bp3 = group.loc[i, ['bp3_target_systolic', 'bp3_target_diastolic']].values.astype(np.float32)
        target_risk = np.array([group.loc[i, 'risk_target']], dtype=np.float32)
        
        data_instances.append({
            'seq_input': seq_input,            # shape (3, feature_dim)
            'static_input': static_input,      # shape (feature_dim,)
            'target_bp24': target_bp24,        # shape (2,)
            'target_bp3': target_bp3,          # shape (2,)
            'target_risk': target_risk         # shape (1,)
        })

print(f"Nombre d'instances : {len(data_instances)}")

# 3. Division en ensembles d'entraînement, validation et test selon les individus
unique_ids = df_model['id_individu'].unique()
train_ids, test_ids = train_test_split(unique_ids, test_size=0.15, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.15/(0.85), random_state=42)

def filter_instances(instances, valid_ids):
    return [inst for inst in instances if inst['seq_input'][0][0] in 
            df_model[df_model['id_individu'].isin(valid_ids)]['id_individu'].values]  # Trick pas parfait...
    
# Pour s'assurer de la répartition par individu, on refait le filtrage à partir des instances:
train_instances = [inst for inst in data_instances if inst['static_input'][0] == inst['static_input'][0]]  # on utilisera le champ 'id_individu' provenant du dataset original
# Mieux : on crée notre Dataset directement en utilisant l'id_individu stocké dans chaque instance.
# On refait la création d'instances en ajoutant l'id_individu :

data_instances = []
for id_indiv, group in df_model.groupby('id_individu'):
    group = group.sort_values('date').reset_index(drop=True)
    if len(group) < 4:
        continue
    for i in range(3, len(group)):
        seq_input = group.loc[i-3:i-1, feature_cols].values  # (3, feature_dim)
        static_input = group.loc[i, feature_cols].values      # (feature_dim,)
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

train_instances = [inst for inst in data_instances if inst['id_individu'] in train_ids]
val_instances = [inst for inst in data_instances if inst['id_individu'] in val_ids]
test_instances = [inst for inst in data_instances if inst['id_individu'] in test_ids]

print(f"Instances Train: {len(train_instances)}, Val: {len(val_instances)}, Test: {len(test_instances)}")

# 4. Définition d'un Dataset PyTorch
class BPTimeSeriesDataset(Dataset):
    def __init__(self, instances):
        self.instances = instances
        
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        inst = self.instances[idx]
        seq_input = torch.tensor(inst['seq_input'], dtype=torch.float)       # (3, feature_dim)
        static_input = torch.tensor(inst['static_input'], dtype=torch.float) # (feature_dim,)
        target_bp24 = torch.tensor(inst['target_bp24'], dtype=torch.float)   # (2,)
        target_bp3 = torch.tensor(inst['target_bp3'], dtype=torch.float)     # (2,)
        target_risk = torch.tensor(inst['target_risk'], dtype=torch.float)   # (1,)
        return seq_input, static_input, target_bp24, target_bp3, target_risk

train_dataset = BPTimeSeriesDataset(train_instances)
val_dataset = BPTimeSeriesDataset(val_instances)
test_dataset = BPTimeSeriesDataset(test_instances)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 5. Instanciation du modèle (MultiTaskBPModel, défini précédemment)
# Supposons que la dimension de chaque feature est len(feature_cols)
seq_input_dim = len(feature_cols)  # pour les mesures horaires
static_input_dim = len(feature_cols)  # pour le vecteur statique

# Importez ou redéfinissez ici le modèle MultiTaskBPModel tel que présenté précédemment.
# (Si le code précédent est dans un autre module, vous pouvez l'importer.)
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
        # seq_input : (batch_size, seq_length, seq_input_dim)
        x = self.seq_proj(seq_input)            # (batch_size, seq_length, d_model)
        x = x.transpose(0, 1)                     # (seq_length, batch_size, d_model)
        x = self.transformer_encoder(x)           # (seq_length, batch_size, d_model)
        x = x.mean(dim=0)                         # (batch_size, d_model)
        fusion = torch.cat([x, static_input], dim=1)  # (batch_size, d_model + static_input_dim)
        bp24_pred = self.bp24_regressor(fusion)
        bp3_pred = self.bp3_regressor(fusion)
        risk_logit = self.risk_classifier(fusion)
        risk_pred = torch.sigmoid(risk_logit)
        return bp24_pred, bp3_pred, risk_pred

model = MultiTaskBPModel(seq_input_dim, static_input_dim, d_model=64, nhead=8, num_layers=2, dropout=0.1)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 6. Définition des fonctions de perte et de l'optimiseur
criterion_reg = nn.MSELoss()
criterion_cls = nn.BCEWithLogitsLoss()  # Pour la classification, on utilisera les logits directement

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 7. Boucle d'entraînement et d'évaluation
num_epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for seq_input, static_input, target_bp24, target_bp3, target_risk in train_loader:
        seq_input = seq_input.to(device)
        static_input = static_input.to(device)
        target_bp24 = target_bp24.to(device)
        target_bp3 = target_bp3.to(device)
        target_risk = target_risk.to(device)
        
        optimizer.zero_grad()
        pred_bp24, pred_bp3, pred_risk = model(seq_input, static_input)
        loss_bp24 = criterion_reg(pred_bp24, target_bp24)
        loss_bp3 = criterion_reg(pred_bp3, target_bp3)
        # Pour la classification, comme le modèle applique déjà sigmoid, si on avait utilisé BCEWithLogitsLoss, il faudrait éviter le sigmoïde. Ici on a déjà transformé avec sigmoid dans le modèle.
        # On peut utiliser BCELoss avec les probabilités.
        criterion_bce = nn.BCELoss()
        loss_risk = criterion_bce(pred_risk, target_risk)
        loss = loss_bp24 + loss_bp3 + loss_risk
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * seq_input.size(0)
    train_loss /= len(train_dataset)
    
    # Évaluation sur l'ensemble de validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for seq_input, static_input, target_bp24, target_bp3, target_risk in val_loader:
            seq_input = seq_input.to(device)
            static_input = static_input.to(device)
            target_bp24 = target_bp24.to(device)
            target_bp3 = target_bp3.to(device)
            target_risk = target_risk.to(device)
            
            pred_bp24, pred_bp3, pred_risk = model(seq_input, static_input)
            loss_bp24 = criterion_reg(pred_bp24, target_bp24)
            loss_bp3 = criterion_reg(pred_bp3, target_bp3)
            loss_risk = criterion_bce(pred_risk, target_risk)
            loss = loss_bp24 + loss_bp3 + loss_risk
            val_loss += loss.item() * seq_input.size(0)
    val_loss /= len(val_dataset)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

# Évaluation finale sur l'ensemble de test
model.eval()
test_loss = 0.0
with torch.no_grad():
    for seq_input, static_input, target_bp24, target_bp3, target_risk in test_loader:
        seq_input = seq_input.to(device)
        static_input = static_input.to(device)
        target_bp24 = target_bp24.to(device)
        target_bp3 = target_bp3.to(device)
        target_risk = target_risk.to(device)
        
        pred_bp24, pred_bp3, pred_risk = model(seq_input, static_input)
        loss_bp24 = criterion_reg(pred_bp24, target_bp24)
        loss_bp3 = criterion_reg(pred_bp3, target_bp3)
        loss_risk = criterion_bce(pred_risk, target_risk)
        loss = loss_bp24 + loss_bp3 + loss_risk
        test_loss += loss.item() * seq_input.size(0)
test_loss /= len(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
