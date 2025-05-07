import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskBPModel(nn.Module):
    def __init__(self, seq_input_dim, static_input_dim, d_model=64, nhead=8, num_layers=2, dropout=0.1):
        super(MultiTaskBPModel, self).__init__()
        # Projection de l'entrée séquentielle vers une dimension adaptée pour le transformer
        self.seq_proj = nn.Linear(seq_input_dim, d_model)
        
        # Définition du Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fusion des représentations du transformer et des données statiques
        fusion_dim = d_model + static_input_dim
        
        # Tête 1 : Prédiction de la TA pour les 24 prochaines heures (systolique et diastolique)
        self.bp24_regressor = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 2 sorties (par ex. systolique, diastolique)
        )
        
        # Tête 2 : Prédiction de l'évolution (delta) de la TA sur 3 jours
        self.bp3_regressor = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 2 sorties : delta systolique et delta diastolique
        )
        
        # Tête 3 : Classification du risque de complication (binaire)
        self.risk_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # sortie logit
        )
    
    def forward(self, seq_input, static_input):
        """
        seq_input: Tensor de forme (batch_size, seq_length, seq_input_dim)
        static_input: Tensor de forme (batch_size, static_input_dim)
        """
        # Projection des données séquentielles
        x = self.seq_proj(seq_input)  # (batch_size, seq_length, d_model)
        
        # Le Transformer Encoder attend un tenseur de forme (seq_length, batch_size, d_model)
        x = x.transpose(0, 1)  # (seq_length, batch_size, d_model)
        x = self.transformer_encoder(x)  # (seq_length, batch_size, d_model)
        
        # Pooling moyen sur la dimension séquentielle
        x = x.mean(dim=0)  # (batch_size, d_model)
        
        # Fusionner avec les données statiques
        fusion = torch.cat([x, static_input], dim=1)  # (batch_size, d_model + static_input_dim)
        
        # Tâches multi-output
        bp24_pred = self.bp24_regressor(fusion)  # Prédiction de la TA pour les 24h (régression)
        bp3_pred = self.bp3_regressor(fusion)    # Prédiction de la variation de TA sur 3 jours (régression)
        risk_logit = self.risk_classifier(fusion)  # Logit pour la classification du risque
        
        # Transformation en probabilité pour la classification
        risk_pred = torch.sigmoid(risk_logit)  # (batch_size, 1)
        
        return bp24_pred, bp3_pred, risk_pred

# Exemple d'utilisation
if __name__ == "__main__":
    # Supposons que chaque mesure séquentielle a 10 features et que le vecteur statique a 5 features
    seq_input_dim = 10
    static_input_dim = 5
    
    model = MultiTaskBPModel(seq_input_dim, static_input_dim, d_model=64, nhead=8, num_layers=2, dropout=0.1)
    
    batch_size = 32
    seq_length = 48  # Par exemple, 48 mesures pour 1 jour
    seq_input = torch.randn(batch_size, seq_length, seq_input_dim)
    static_input = torch.randn(batch_size, static_input_dim)
    
    bp24_pred, bp3_pred, risk_pred = model(seq_input, static_input)
    print("Prédiction TA 24h shape:", bp24_pred.shape)         # (batch_size, 2)
    print("Prédiction delta TA sur 3 jours shape:", bp3_pred.shape)  # (batch_size, 2)
    print("Prédiction de risque shape:", risk_pred.shape)        # (batch_size, 1)
