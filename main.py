# main.py

# --- 1. Imports et configuration ----------------------------------------
import os
from datetime import date, datetime
from typing import List, Optional, Literal

from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, confloat, conint, validator

import torch
import json as _json
import pickle
import numpy as np

from sqlalchemy import (
    create_engine, Column, Integer, Float, Boolean, Date, String, ForeignKey, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# --- 2. Base de données -------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:pass@db:5432/yourdb"
)

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def get_db():
    """
    Ouvre une session de base de données SQLAlchemy et la ferme ensuite.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 3. ORM Models ------------------------------------------------------
class User(Base):
    __tablename__ = "users"
    id_individu            = Column(Integer, primary_key=True, index=True)
    dob                    = Column(Date, nullable=False)
    sex                    = Column(String, nullable=False)
    weight_kg              = Column(Float, nullable=False)
    height_cm              = Column(Float, nullable=False)
    htn_diagnosed          = Column(Boolean, nullable=False)
    other_cv_history       = Column(String, nullable=True)
    diabetes               = Column(Boolean, default=False)
    dyslipidemia           = Column(Boolean, default=False)
    renal_failure          = Column(Boolean, default=False)
    heart_failure          = Column(Boolean, default=False)
    smoking_status         = Column(String, nullable=False)
    cig_per_day            = Column(Integer, default=0)
    alcohol_units_per_week = Column(Integer, default=0)
    activity_level         = Column(String, nullable=False)
    sleep_hours            = Column(Float, default=0)
    adherence_pct          = Column(Integer, nullable=False)
    family_hta             = Column(Boolean, nullable=False)
    family_cv_early        = Column(Boolean, nullable=False)
    last_physician_visit   = Column(Date, nullable=True)
    bp_target_sys          = Column(Integer, nullable=True)
    bp_target_dia          = Column(Integer, nullable=True)
    pregnancy_status       = Column(String, nullable=True)
    menopause              = Column(Boolean, nullable=True)
    antihypertensive_rx    = Column(JSON, nullable=True)

class DailyMeasurement(Base):
    __tablename__ = "daily_measurements"
    id                      = Column(Integer, primary_key=True, index=True)
    id_individu             = Column(Integer, ForeignKey("users.id_individu"), index=True)
    date                    = Column(Date, nullable=False)
    ta_systolique_mean      = Column(Float, nullable=False)
    ta_systolique_std       = Column(Float, nullable=False)
    ta_systolique_min       = Column(Float, nullable=False)
    ta_systolique_max       = Column(Float, nullable=False)
    ta_diastolique_mean     = Column(Float, nullable=False)
    ta_diastolique_std      = Column(Float, nullable=False)
    ta_diastolique_min      = Column(Float, nullable=False)
    ta_diastolique_max      = Column(Float, nullable=False)
    heart_rate_mean         = Column(Float, nullable=False)
    heart_rate_std          = Column(Float, nullable=False)
    stress_level_mean       = Column(Float, nullable=False)
    steps_sum               = Column(Float, nullable=False)
    temperature_mean        = Column(Float, nullable=False)
    temperature_std         = Column(Float, nullable=False)
    humidity_mean           = Column(Float, nullable=False)
    humidity_std            = Column(Float, nullable=False)
    complication_event_max  = Column(Integer, nullable=False)
    ta_systolic_delta       = Column(Float, nullable=False)
    ta_diastolic_delta      = Column(Float, nullable=False)
    day_of_week             = Column(String, nullable=False)
    treatment_type          = Column(String, nullable=False)
    bp_evolution_next3days  = Column(String, nullable=False)

# Crée les tables si nécessaire
Base.metadata.create_all(engine)

# --- 4. Pydantic Schemas ------------------------------------------------
class Drug(BaseModel):
    name: str
    dose_mg: confloat(gt=0)

class FamilyHistoryIn(BaseModel):
    hta_early: bool = Field(..., alias="hta_early")
    cv_event_early: bool = Field(..., alias="cv_event_early")
    class Config:
        allow_population_by_field_name = True

class UserProfile(BaseModel):
    user_id: int = Field(..., alias="user_id")
    dob: date
    sex: Literal["M", "F", "Other"]
    weight_kg: confloat(gt=0)
    height_cm: confloat(gt=0)
    htn_diagnosed: bool
    other_cv_history: List[str] = []
    diabetes: bool = False
    dyslipidemia: bool = False
    renal_failure: bool = False
    heart_failure: bool = False
    antihypertensive_rx: List[Drug] = []
    smoking_status: Literal["none", "current", "former"]
    cig_per_day: conint(ge=0) = 0
    alcohol_units_per_week: conint(ge=0) = 0
    activity_level: Literal["sedentary", "moderate", "intense"]
    sleep_hours: confloat(ge=0) = 0
    adherence_pct: conint(ge=0, le=100)
    family_history: FamilyHistoryIn
    last_physician_visit: Optional[date] = None
    bp_target_sys: Optional[conint(gt=0)] = None
    bp_target_dia: Optional[conint(gt=0)] = None
    pregnancy_status: Optional[Literal["pregnant", "postpartum"]] = None
    menopause: Optional[bool] = None

    @validator("dob")
    def check_age(cls, v):
        age = (datetime.today().date() - v).days / 365
        if not (0 < age < 120):
            raise ValueError("Âge non valide")
        return v

    class Config:
        allow_population_by_field_name = True

class Measurement(BaseModel):
    id_individu: int
    date: date
    ta_systolique_mean: float
    ta_systolique_std: float
    ta_systolique_min: float
    ta_systolique_max: float
    ta_diastolique_mean: float
    ta_diastolique_std: float
    ta_diastolique_min: float
    ta_diastolique_max: float
    heart_rate_mean: float
    heart_rate_std: float
    stress_level_mean: float
    steps_sum: float
    temperature_mean: float
    temperature_std: float
    humidity_mean: float
    humidity_std: float
    complication_event_max: int
    ta_systolic_delta: float
    ta_diastolic_delta: float
    day_of_week: Literal[
        "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"
    ]
    treatment_type: Literal[
        "aucun","eau","médicament","médicament + activité sportive",
        "médicament + repos","médicament + repos + activité sportive",
        "repos","repos + eau","voir le médecin","voir le médecin + médicament"
    ]
    bp_evolution_next3days: Literal[
        "amélioration significative","aucune","stabilité"
    ]

class FamilyHistoryOut(BaseModel):
    hta_early: bool
    cv_event_early: bool
    class Config:
        orm_mode = True

class UserOut(BaseModel):
    id_individu: int
    dob: date
    sex: str
    weight_kg: float
    height_cm: float
    htn_diagnosed: bool
    other_cv_history: List[str]
    diabetes: bool
    dyslipidemia: bool
    renal_failure: bool
    heart_failure: bool
    smoking_status: str
    cig_per_day: int
    alcohol_units_per_week: int
    activity_level: str
    sleep_hours: float
    adherence_pct: int
    family_history: FamilyHistoryOut
    last_physician_visit: Optional[date]
    bp_target_sys: Optional[int]
    bp_target_dia: Optional[int]
    pregnancy_status: Optional[str]
    menopause: Optional[bool]
    antihypertensive_rx: List[Drug]
    class Config:
        orm_mode = True

class DailyMeasurementOut(BaseModel):
    id: int
    id_individu: int
    date: date
    ta_systolique_mean: float
    ta_systolique_std: float
    ta_systolique_min: float
    ta_systolique_max: float
    ta_diastolique_mean: float
    ta_diastolique_std: float
    ta_diastolique_min: float
    ta_diastolique_max: float
    heart_rate_mean: float
    heart_rate_std: float
    stress_level_mean: float
    steps_sum: float
    temperature_mean: float
    temperature_std: float
    humidity_mean: float
    humidity_std: float
    complication_event_max: int
    ta_systolic_delta: float
    ta_diastolic_delta: float
    day_of_week: str
    treatment_type: str
    bp_evolution_next3days: str
    class Config:
        orm_mode = True

# --- 5. Chargement du modèle & resources -------------------------------
with open("feature_cols.json", "r") as f:
    feature_cols = _json.load(f)
with open("scaler_bp.pkl", "rb") as f:
    target_scaler = pickle.load(f)

class MultiTaskBPModel(torch.nn.Module):
    def __init__(self, seq_dim, static_dim, d_model=64, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.seq_proj = torch.nn.Linear(seq_dim, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        fusion_dim = d_model + static_dim
        self.bp24_regressor = torch.nn.Sequential(
            torch.nn.Linear(fusion_dim, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
        )
        self.bp3_regressor = torch.nn.Sequential(
            torch.nn.Linear(fusion_dim, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
        )
        self.risk_classifier = torch.nn.Sequential(
            torch.nn.Linear(fusion_dim, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)
        )

    def forward(self, seq_input, static_input):
        x = self.seq_proj(seq_input)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        fusion = torch.cat([x, static_input], dim=1)
        return (
            self.bp24_regressor(fusion),
            self.bp3_regressor(fusion),
            self.risk_classifier(fusion),
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskBPModel(len(feature_cols), len(feature_cols))
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device).eval()

SEX_MAP = {"M": 0, "F": 1, "Other": 2}

# --- 6. FastAPI ----------------------------------------------------------
app = FastAPI(
    title="BP Prediction API",
    description="API pour enregistrer des utilisateurs, ingérer des mesures journalières et prédire l'évolution de la tension artérielle."
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre en production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post(
    "/profile",
    response_model=dict,
    status_code=201,
    summary="Créer / Mettre à jour un profil utilisateur",
    description="Reçoit un payload `UserProfile` et crée ou met à jour l'utilisateur dans la base."
)
def set_profile(
    profile: UserProfile = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    data = profile.dict(by_alias=True)
    data["antihypertensive_rx"] = [d.dict() for d in profile.antihypertensive_rx]
    data["other_cv_history"]    = ",".join(data.get("other_cv_history", []))
    data["family_hta"]          = data["family_history"]["hta_early"]
    data["family_cv_early"]     = data["family_history"]["cv_event_early"]
    del data["family_history"]

    uid = data.pop("user_id")
    user = db.query(User).get(uid)
    if user:
        for k, v in data.items():
            setattr(user, k, v)
    else:
        user = User(id_individu=uid, **data)
        db.add(user)
    db.commit()
    return {"status": "profil enregistré"}

@app.get(
    "/profile/{id_individu}",
    response_model=UserOut,
    summary="Récupérer un profil",
    description="Retourne les informations de l'utilisateur d'ID donné."
)
def get_profile(id_individu: int, db: Session = Depends(get_db)):
    user = db.query(User).get(id_individu)
    if not user:
        raise HTTPException(status_code=404, detail="Profil inexistant")
    oc  = user.other_cv_history.split(",") if user.other_cv_history else []
    fam = FamilyHistoryOut(hta_early=user.family_hta, cv_event_early=user.family_cv_early)
    return UserOut(
        id_individu=user.id_individu,
        dob=user.dob,
        sex=user.sex,
        weight_kg=user.weight_kg,
        height_cm=user.height_cm,
        htn_diagnosed=user.htn_diagnosed,
        other_cv_history=oc,
        diabetes=user.diabetes,
        dyslipidemia=user.dyslipidemia,
        renal_failure=user.renal_failure,
        heart_failure=user.heart_failure,
        smoking_status=user.smoking_status,
        cig_per_day=user.cig_per_day,
        alcohol_units_per_week=user.alcohol_units_per_week,
        activity_level=user.activity_level,
        sleep_hours=user.sleep_hours,
        adherence_pct=user.adherence_pct,
        family_history=fam,
        last_physician_visit=user.last_physician_visit,
        bp_target_sys=user.bp_target_sys,
        bp_target_dia=user.bp_target_dia,
        pregnancy_status=user.pregnancy_status,
        menopause=user.menopause,
        antihypertensive_rx=user.antihypertensive_rx or []
    )

@app.get(
    "/profiles",
    response_model=List[UserOut],
    summary="Lister tous les profils",
    description="Retourne la liste de tous les utilisateurs enregistrés."
)
def list_profiles(db: Session = Depends(get_db)):
    users = db.query(User).all()
    result = []
    for u in users:
        oc  = u.other_cv_history.split(",") if u.other_cv_history else []
        fam = FamilyHistoryOut(hta_early=u.family_hta, cv_event_early=u.family_cv_early)
        result.append(UserOut(
            id_individu=u.id_individu,
            dob=u.dob,
            sex=u.sex,
            weight_kg=u.weight_kg,
            height_cm=u.height_cm,
            htn_diagnosed=u.htn_diagnosed,
            other_cv_history=oc,
            diabetes=u.diabetes,
            dyslipidemia=u.dyslipidemia,
            renal_failure=u.renal_failure,
            heart_failure=u.heart_failure,
            smoking_status=u.smoking_status,
            cig_per_day=u.cig_per_day,
            alcohol_units_per_week=u.alcohol_units_per_week,
            activity_level=u.activity_level,
            sleep_hours=u.sleep_hours,
            adherence_pct=u.adherence_pct,
            family_history=fam,
            last_physician_visit=u.last_physician_visit,
            bp_target_sys=u.bp_target_sys,
            bp_target_dia=u.bp_target_dia,
            pregnancy_status=u.pregnancy_status,
            menopause=u.menopause,
            antihypertensive_rx=u.antihypertensive_rx or []
        ))
    return result

@app.post(
    "/measurement",
    response_model=dict,
    summary="Ingestion de mesure journalière",
    description="Reçoit un `Measurement` et l'enregistre en base."
)
def ingest(meas: Measurement, db: Session = Depends(get_db)):
    if not db.query(User).get(meas.id_individu):
        raise HTTPException(status_code=404, detail="Profil inexistant")
    dm = DailyMeasurement(**meas.dict())
    db.add(dm)
    db.commit()
    return {"status": "mesure ingérée"}

@app.get(
    "/measurements",
    response_model=List[DailyMeasurementOut],
    summary="Lister toutes les mesures",
    description="Retourne la liste de toutes les mesures journalières."
)
def list_measurements(db: Session = Depends(get_db)):
    return db.query(DailyMeasurement).all()

@app.get(
    "/measurements/{measurement_id}",
    response_model=DailyMeasurementOut,
    summary="Récupérer une mesure",
    description="Retourne la mesure dont l'ID est fourni."
)
def get_measurement(measurement_id: int, db: Session = Depends(get_db)):
    dm = db.query(DailyMeasurement).get(measurement_id)
    if not dm:
        raise HTTPException(status_code=404, detail="Mesure non trouvée")
    return dm

@app.get(
    "/users/{id_individu}/measurements",
    response_model=List[DailyMeasurementOut],
    summary="Mesures d'un utilisateur",
    description="Retourne toutes les mesures journalières pour l'utilisateur donné."
)
def get_user_measurements(id_individu: int, db: Session = Depends(get_db)):
    return db.query(DailyMeasurement).filter_by(id_individu=id_individu).all()

@app.get(
    "/predict/{id_individu}",
    summary="Prédiction BP",
    description=(
        "Génère la prédiction de la tension artérielle pour les 24h et 3 prochains jours, "
        "et la probabilité de risque à partir des 3 derniers jours de mesures."
    )
)
def predict(id_individu: int, db: Session = Depends(get_db)):
    # Validation du profil
    user = db.query(User).get(id_individu)
    if not user:
        raise HTTPException(status_code=404, detail="Profil inexistant")

    # Récupère les 3 dernières journées de mesures
    rows = (
        db.query(DailyMeasurement)
          .filter(DailyMeasurement.id_individu == id_individu)
          .order_by(DailyMeasurement.date.desc())
          .limit(3)
          .all()
    )
    if len(rows) < 3:
        raise HTTPException(status_code=400, detail="3 jours requis")

    # Construit la séquence d'entrée
    seq = []
    for dm in reversed(rows):
        base = dm.__dict__.copy()
        age = int((datetime.today().date() - user.dob).days / 365)
        IMC = round(user.weight_kg / (user.height_cm / 100)**2, 1)
        base.update({
            "age": age,
            "sexe": SEX_MAP[user.sex],
            "antécédents": int(user.htn_diagnosed),
            "poids": user.weight_kg,
            "taille": user.height_cm,
            "IMC": IMC
        })
        for col in feature_cols:
            if col.startswith("day_of_week_"):
                base[col] = 1 if dm.day_of_week == col.split("_",2)[2] else 0
            elif col.startswith("treatment_type_"):
                val = dm.treatment_type
                base[col] = 1 if val == col.replace("treatment_type_","") else 0
            elif col.startswith("bp_evolution_next3days_"):
                val = dm.bp_evolution_next3days
                base[col] = 1 if val == col.replace("bp_evolution_next3days_","") else 0
        seq.append([base.get(c,0) for c in feature_cols])

    arr    = torch.tensor(np.array(seq), dtype=torch.float32).unsqueeze(0).to(device)
    static = torch.tensor(np.array(seq[-1]), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        bp24_p, bp3_p, risk_l = model(arr, static)
        risk_p = torch.sigmoid(risk_l).cpu().numpy().tolist()

        # Remise à l'échelle des prédictions
        bp24_scaled = bp24_p.cpu().numpy()
        bp3_scaled  = bp3_p.cpu().numpy()
        bp24 = target_scaler.inverse_transform(bp24_scaled)
        bp3  = target_scaler.inverse_transform(bp3_scaled)

        # Clamp physiologique et sys ≥ dia
        bp24[:,0] = np.clip(bp24[:,0], 50,250); bp24[:,1] = np.clip(bp24[:,1],30,150)
        bp3[:,0]  = np.clip(bp3[:,0], 50,250);  bp3[:,1]  = np.clip(bp3[:,1],30,150)
        for arr_ in (bp24,bp3):
            if arr_[0,0] < arr_[0,1]:
                arr_[0,1] = arr_[0,0]

        return {
            "bp24_prediction": bp24.tolist(),
            "bp3_prediction": bp3.tolist(),
            "risk_probability": risk_p
        }

@app.get(
    "/metadata/features",
    summary="Liste des features",
    description="Retourne la liste des noms de colonnes utilisées comme features dans le modèle."
)
def get_features():
    return {"feature_cols": feature_cols}

@app.get(
    "/metadata/enums",
    summary="Valeurs autorisées pour les énumérations",
    description="Retourne les choix valides pour les différents champs de type énuméré."
)
def get_enums():
    return {
        "sex": ["M","F","Other"],
        "activity_level": ["sedentary","moderate","intense"],
        "smoking_status": ["none","current","former"],
        "treatment_type": [
            "aucun","eau","médicament","médicament + activité sportive",
            "médicament + repos","médicament + repos + activité sportive",
            "repos","repos + eau","voir le médecin","voir le médecin + médicament"
        ],
        "bp_evolution_next3days": ["amélioration significative","aucune","stabilité"],
        "days_of_week": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    }

@app.get(
    "/metadata/model",
    summary="Infos sur le modèle",
    description="Retourne les dimensions du modèle (seq_dim, static_dim, d_model, n_layers)."
)
def get_model_info():
    return {
        "seq_dim": len(feature_cols),
        "static_dim": len(feature_cols),
        "d_model": model.seq_proj.out_features,
        "n_layers": len(model.transformer_encoder.layers)
    }
