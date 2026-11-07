"""Engenharia de features.

- horizonte.py: dataset multi-horizonte (t0 → t0+1..t0+12), usado pela previsão de 12 meses;
- sazonal.py: features do modelo sazonal antigo (um mês, sem horizonte), mantidas por compatibilidade.
"""
from algamar_ml.features.sazonal import (  # noqa: F401  (compatibilidade com os scripts antigos)
    CHAVE, CHAVE_PONTO, FEATURES_BASELINE, HOTSPOTS, LIMITE_FLORACAO,
    climatologia_para_previsao, climatologia_passada, confere_chave_unica, constroi_features,
    constroi_features_original, distancia_hotspot, indice_mensal, lag_por_data, media_movel_passada,
)
