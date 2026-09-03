import copernicusmarine

print("Iniciando conexão com Copernicus Marine...")

copernicusmarine.subset(
    dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
    variables=["CHL"],
    minimum_longitude=-46.0,
    maximum_longitude=-42.0,
    minimum_latitude=-25.0,
    maximum_latitude=-22.0,
    start_datetime="2024-01-01T00:00:00",
    end_datetime="2024-01-07T00:00:00",
    output_filename="teste_clorofila.nc",
    output_directory="dados",
    overwrite=True
)

print("Download concluído com sucesso!")