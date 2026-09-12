import copernicusmarine

regiao_correntes = {
    "minimum_longitude": -47.5,
    "maximum_longitude": -44.3,
    "minimum_latitude": -25.8,
    "maximum_latitude": -22.8,
}

anos = list(range(2000, 2025))

for ano in anos:
    print(f"\n===== Ano {ano} — Correntes Marítimas =====")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1M-m",
        variables=["uo", "vo"],
        **regiao_correntes,
        minimum_depth=0,
        maximum_depth=1,
        start_datetime=f"{ano}-01-01T00:00:00",
        end_datetime=f"{ano}-12-31T00:00:00",
        output_filename=f"correntes_{ano}_sp.nc",
        output_directory="dados",
        overwrite=True
    )
    print(f"Correntes {ano} concluído!")

print("\n\nTodos os anos de correntes marítimas baixados com sucesso!")