"""Baixa do OBIS (API v3) as ocorrências das espécies tóxicas no litoral de RJ a SC.  Uso: python baixar_obis.py dados/obis"""
import json, sys, time, urllib.parse, urllib.request
from pathlib import Path

DEST = Path(sys.argv[1])
TAXA = ["Dinophysis", "Alexandrium", "Gymnodinium catenatum", "Pseudo-nitzschia"]
# RJ (Cabo Frio) a SC, faixa costeira e plataforma
GEOM = "POLYGON((-49.5 -29.5, -40.5 -29.5, -40.5 -22.0, -49.5 -22.0, -49.5 -29.5))"
CAMPOS = "id,scientificName,eventDate,date_year,month,decimalLatitude,decimalLongitude,depth,individualCount,organismQuantity,organismQuantityType,datasetName,basisOfRecord,occurrenceStatus"

def pega(nome):
    regs, after = [], None
    while True:
        q = {"scientificname": nome, "geometry": GEOM, "size": 5000, "fields": CAMPOS}
        if after: q["after"] = after
        url = "https://api.obis.org/v3/occurrence?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "AlgaMar-TCC/1.0"}), timeout=120) as r:
            d = json.load(r)
        res = d.get("results", [])
        regs += res
        if len(res) < 5000: return regs, d.get("total")
        after = res[-1]["id"]; time.sleep(1)

for t in TAXA:
    regs, total = pega(t)
    (DEST / f"{t.replace(' ', '_')}.json").write_text(json.dumps(regs, ensure_ascii=False), encoding="utf-8")
    print(f"{t}: {len(regs)} registros (total OBIS {total})")
