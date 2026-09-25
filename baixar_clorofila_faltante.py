import copernicusmarine

regiao = {
    "minimum_longitude": -47.0,
    "maximum_longitude": -44.8,
    "minimum_latitude": -25.3,
    "maximum_latitude": -23.3,
}

anos_faltantes = [2000]  # ajuste aqui se faltar mais de um ano

for ano in anos_faltantes:
    print(f"\n===== Ano {ano} — Clorofila =====")
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
    print(f"Clorofila {ano} concluída!")

print("\nConcluído!")