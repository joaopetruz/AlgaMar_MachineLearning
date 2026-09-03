import copernicusmarine

print("Baixando dados de temperatura...")

copernicusmarine.subset(
    dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
    variables=["analysed_sst"],
    minimum_longitude=-46.0,
    maximum_longitude=-42.0,
    minimum_latitude=-25.0,
    maximum_latitude=-22.0,
    start_datetime="2024-01-01T00:00:00",
    end_datetime="2024-01-07T00:00:00",
    output_filename="teste_temperatura.nc",
    output_directory="dados",
    overwrite=True
)

print("Download concluído com sucesso!")