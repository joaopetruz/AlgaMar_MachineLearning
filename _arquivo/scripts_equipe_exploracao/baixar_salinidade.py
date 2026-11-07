import copernicusmarine

regiao_sal = {
    "minimum_longitude": -47.5,
    "maximum_longitude": -44.3,
    "minimum_latitude": -25.8,
    "maximum_latitude": -22.8,
}

anos = list(range(2000, 2025))

for ano in anos:
    print(f"\n===== Ano {ano} — Salinidade =====")
    copernicusmarine.subset(
        dataset_id="cmems_obs-mob_glo_phy-sss_my_multi_P1D",
        variables=["sos"],  # sos = sea surface salinity
        **regiao_sal,
        start_datetime=f"{ano}-01-01T00:00:00",
        end_datetime=f"{ano}-12-31T00:00:00",
        output_filename=f"salinidade_{ano}_sp.nc",
        output_directory="dados",
        overwrite=True
    )
    print(f"Salinidade {ano} concluída!")

print("\n\nTodos os anos de salinidade baixados com sucesso!")