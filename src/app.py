import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest

carpeta_datos = "./datos"
os.makedirs(carpeta_datos, exist_ok=True)
enlace = "https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv"
ruta_datos = os.path.join(carpeta_datos, "AB_NYC_2019.csv")

df = pd.read_csv(enlace)
df.to_csv(ruta_datos, index=False)

# Limpio los na
df['ultima_resena'] = df['last_review'].fillna('Sin reseñas')
df['resenas_por_mes'] = df['reviews_per_month'].fillna(0)
df = df.drop_duplicates()

# Filtro por si hay precios fuera de los cuartiles
q1, q3 = df['price'].quantile([0.25, 0.75])
iqr = q3 - q1
limite_inferior, limite_superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df = df[(df['price'] >= limite_inferior) & (df['price'] <= limite_superior) & (df['price'] > 0)]

# Si hay cosas anomalas
q1, q3 = df['minimum_nights'].quantile([0.25, 0.75])
iqr = q3 - q1
limite_inferior, limite_superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df = df[(df['minimum_nights'] >= limite_inferior) & (df['minimum_nights'] <= limite_superior) & (df['minimum_nights'] > 0)]

# Si hay columnas innecesarias
df.drop(["id", "name", "host_id", "host_name", "last_review", "reviews_per_month", "latitude", "longitude", "neighbourhood"], axis=1, inplace=True)

# Nueva columna: precio por noche
df['precio_por_noche'] = df['price'] / df['minimum_nights']

# Ponemos precios
def clasificar_precio(precio):
    if precio < 50:
        return "Bajo"
    elif precio < 150:
        return "Medio"
    else:
        return "Alto"

df['categoria_precio'] = df['price'].apply(clasificar_precio)
df["tipo_habitacion"] = pd.factorize(df["room_type"])[0]
df["grupo_vecindario"] = pd.factorize(df["neighbourhood_group"])[0]
df.drop("room_type", axis=1, inplace=True)
df.drop("neighbourhood_group", axis=1, inplace=True)

# Normalizado
columnas_numericas = ["number_of_reviews", "minimum_nights", "calculated_host_listings_count", "availability_365", "tipo_habitacion", "grupo_vecindario", "precio_por_noche"]

escalador = MinMaxScaler()
df_escalado = pd.DataFrame(escalador.fit_transform(df[columnas_numericas]), columns=columnas_numericas, index=df.index)
df_escalado["precio"] = df["price"]



# Separar datos
X = df_escalado.drop("precio", axis=1)
y = df_escalado["precio"]

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Eleccion
modelo_seleccion = SelectKBest(chi2, k=4)
modelo_seleccion.fit(X_entrenamiento, y_entrenamiento)
indices = modelo_seleccion.get_support()
X_entrenamiento_sel = pd.DataFrame(modelo_seleccion.transform(X_entrenamiento), columns=X_entrenamiento.columns.values[indices])
X_prueba_sel = pd.DataFrame(modelo_seleccion.transform(X_prueba), columns=X_prueba.columns.values[indices])

X_entrenamiento_sel["precio"] = list(y_entrenamiento)
X_prueba_sel["precio"] = list(y_prueba)

# Guardamos los datos!
carpeta_procesados = "./datos_procesados"
os.makedirs(carpeta_procesados, exist_ok=True)

X_entrenamiento_sel.to_csv(os.path.join(carpeta_procesados, "entrenamiento.csv"), index=False)
X_prueba_sel.to_csv(os.path.join(carpeta_procesados, "prueba.csv"), index=False)