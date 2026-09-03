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

# Anos que vamos baixar (ajuste conforme necessário)
anos = list(range(2000, 2025))  # 2000 a 2024 = 25 anos

for ano in anos:
    print(f"\n===== Ano {ano} =====")

    print(f"Baixando clorofila-a {ano}...")
    copernicusmarine.subset(
        dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
        variables=["CHL"],
        **regiao,
        start_datetime=f"{ano}-01-01T00:00:00",
        end_datetime=f"{ano}-12-31T00:00:00",
        output_filename=f"clorofila_{ano}_sp.nc",
        output_directory="dados",
        overwrite=True
    )
    print(f"Clorofila-a {ano} concluída!")

    print(f"Baixando temperatura {ano}...")
    copernicusmarine.subset(
        dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
        variables=["analysed_sst"],
        **regiao_temp,
        start_datetime=f"{ano}-01-01T00:00:00",
        end_datetime=f"{ano}-12-31T00:00:00",
        output_filename=f"temperatura_{ano}_sp.nc",
        output_directory="dados",
        overwrite=True
    )
    print(f"Temperatura {ano} concluída!")

print("\n\nTodos os anos baixados com sucesso!")