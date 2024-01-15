import pandas as pd
from datasets import Dataset
import json

# Paso 1: Leer el archivo JSON
with open('C:/Users/jesus/Desktop/23_24/NN/xVal_main/xVal_main/data/test.json') as f:
    data = json.load(f)

# Paso 2: Convertir el JSON a DataFrame de Pandas
# Crear una lista vacía para almacenar los datos
data_list = []

# Iterar sobre cada elemento en "features"
for feature in data["features"]:
    # Extraer las propiedades y las coordenadas
    properties = feature["properties"]
    coordinates = feature["geometry"]["coordinates"]
    properties["Longitude"] = coordinates[0]
    properties["Latitude"] = coordinates[1]
    # Añadir al listado
    data_list.append(properties)

# Convertir el listado en DataFrame
df = pd.DataFrame(data_list)

# Paso 3: Convertir el DataFrame a un Dataset de HuggingFace
ds = Dataset.from_pandas(df)

# Ver el conjunto de datos
print(ds)
