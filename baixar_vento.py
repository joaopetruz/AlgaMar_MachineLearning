import copernicusmarine

regiao_vento = {
    "minimum_longitude": -47.5,
    "maximum_longitude": -44.3,
    "minimum_latitude": -25.8,
    "maximum_latitude": -22.8,
}

anos = list(range(2000, 2025))

for ano in anos:
    print(f"\n===== Ano {ano} — Vento =====")
    copernicusmarine.subset(
        dataset_id="cmems_obs-wind_glo_phy_my_l4_P1M",
        variables=["eastward_wind", "northward_wind", "wind_speed"],
        **regiao_vento,
        start_datetime=f"{ano}-01-01T00:00:00",
        end_datetime=f"{ano}-12-31T00:00:00",
        output_filename=f"vento_{ano}_sp.nc",
        output_directory="dados",
        overwrite=True
    )
    print(f"Vento {ano} concluído!")

print("\n\nTodos os anos de vento baixados com sucesso!")