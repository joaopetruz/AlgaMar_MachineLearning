import copernicusmarine

regiao = {
    "minimum_longitude": -47.0,
    "maximum_longitude": -44.8,
    "minimum_latitude": -25.3,
    "maximum_latitude": -23.3,
}

regiao_temp = {
    "minimum_longitude": -47.5,
    "maximum_longitude": -44.3,
    "minimum_latitude": -25.8,
    "maximum_latitude": -22.8,
}

periodo = {
    "start_datetime": "2024-01-01T00:00:00",
    "end_datetime": "2024-12-31T00:00:00",
}

print("Baixando clorofila-a (litoral SP)...")
copernicusmarine.subset(
    dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
    variables=["CHL"],
    **regiao,
    **periodo,
    output_filename="clorofila_2024_sp.nc",
    output_directory="dados",
    overwrite=True
)
print("Clorofila-a concluída!\n")

print("Baixando temperatura (litoral SP)...")
copernicusmarine.subset(
    dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
    variables=["analysed_sst"],
    **regiao_temp,
    **periodo,
    output_filename="temperatura_2024_sp.nc",
    output_directory="dados",
    overwrite=True
)
print("Temperatura concluída!\n")
print("Tudo pronto!")