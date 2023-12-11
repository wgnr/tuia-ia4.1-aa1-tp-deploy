from joblib import load
import numpy as np
import pandas as pd
import streamlit as st

caratula = """
---

# Aprendizaje Automatico 1

## Tecnicatura Universitaria en Inteligencia Artificial

### **Materia: Aprendizaje Automático 1 (IA4.1)**

**Trabajo Práctico: Predicción de lluvia en Australia**

**Docentes:**
- Agustín Almada
- Bruno Cocitto
- Joel Spak

**Integrantes:**
| Apellido y Nombre | Legajo   |
| --- | --- |
| Palermo, Leonel    | P-5192/6 |
| Wagner, Juan       | W-0557/6 |


Año: 2023

---
"""

descripcion_tp = """El presente trabajo práctico consiste en la implementación de un modelo de aprendizaje automático que permita predecir si lloverá o no en Australia al día siguiente. Para ello se utilizará un dataset de registros meteorológicos de los últimos 10 años, el cual contiene información sobre la temperatura, la humedad, la presión, la velocidad y dirección del viento, la nubosidad, la cantidad de lluvia y la evaporación, entre otros datos.

Se ponen a disposicion todas las caracterísiticas del dataset para que el usuario pueda ajustar los valores de las mismas y obtener la predicción de los modelos implementados.


"""


# CAPA DATOS
def load_columns(path):
    with open(path, "r") as f:
        return f.read().split(";")


modelos = {
    "Dummy Base Models": {
        "regresion": {
            "model": load("modelos/model_reg_dummy.pkl"),
            "columns_x_train": [],
        },
        "clasificacion": {
            "model": load("modelos/model_clf_dummy.pkl"),
            "columns_x_train": [],
        },
    },
    "scikit-learn Models": {
        "regresion": {
            "model": load("modelos/model_reg_ml.pkl"),
            "columns_x_train": load_columns("modelos/model_reg_ml_columns_x_train.txt"),
        },
        "clasificacion": {
            "model": load("modelos/model_clf_ml.pkl"),
            "columns_x_train": load_columns("modelos/model_clf_ml_columns_x_train.txt"),
        },
    },
    "TensorFlow NN": {
        "regresion": {
            "model": load("modelos/model_reg_nn.pkl"),
            "columns_x_train": load_columns("modelos/model_reg_nn_columns_x_train.txt"),
        },
        "clasificacion": {
            "model": load("modelos/model_clf_nn.pkl"),
            "columns_x_train": load_columns("modelos/model_clf_nn_columns_x_train.txt"),
        },
    },
    "TensorFlow NN + Optuna": {
        "regresion": {
            "model": load("modelos/model_reg_nn_optuna.pkl"),
            "columns_x_train": load_columns(
                "modelos/model_reg_nn_optuna_columns_x_train.txt"
            ),
        },
        "clasificacion": {
            "model": load("modelos/model_clf_nn_optuna.pkl"),
            "columns_x_train": load_columns(
                "modelos/model_clf_nn_optuna_columns_x_train.txt"
            ),
        },
    },
}

df = pd.read_json("modelos/summary.json")
columns_of_interest = [
    ("Generales", ["Location", "RainToday", "Rainfall", "Evaporation", "Sunshine"]),
    ("Temperatura", ["MaxTemp", "MinTemp", "Temp3pm", "Temp9am"]),
    ("Nubosidad", ["Cloud3pm", "Cloud9am"]),
    ("Presion", ["Pressure3pm", "Pressure9am"]),
    ("Humedad", ["Humidity3pm", "Humidity9am"]),
    ("Velocidad del Viento", ["WindGustSpeed", "WindSpeed3pm", "WindSpeed9am"]),
    ("Direccion del Viento", ["WindDir3pm", "WindDir9am", "WindGustDir"]),
]


# CAPA VISTA
st.set_page_config(
    page_title="Llueve sobre mojado",
    page_icon=":sun_with_face:",
    layout="wide",
)


st.markdown(caratula)

st.title("Predicción de lluvia en Australia")

st.markdown(descripcion_tp)


# Variable para guardar todos los valores de los inputs
values = {}

# Armamos todas las botoneras en forma dinámica
st.header("Condiciones Meteorologicas del Día Anterior")
cols = st.columns(len(columns_of_interest))
for i, (header_titlte, col_group) in enumerate(columns_of_interest):
    cols[i].subheader(header_titlte)
    for col_name in sorted(col_group):
        if pd.api.types.is_numeric_dtype(df[col_name]):
            max_v = df.loc["max", col_name]
            min_v = df.loc["min", col_name]
            values[col_name] = cols[i].slider(
                label=col_name,
                help=f"{min_v} <= {col_name} <= {max_v}",
                value=df.loc["mean", col_name],
                min_value=min_v,
                max_value=max_v,
            )
        else:
            unique_values = df.loc["unique_values", col_name].split(";")
            values[col_name] = cols[i].selectbox(
                label=col_name,
                options=unique_values,
                index=unique_values.index(df.loc["top", col_name]),
            )


def eval_models(values):
    result = {}
    for model_name in modelos:
        result[model_name] = {}
        for model_type in modelos[model_name]:
            model = modelos[model_name][model_type]["model"]
            columns_x_train = modelos[model_name][model_type]["columns_x_train"]
            x_test = []
            for col in columns_x_train:
                if "_" not in col:
                    x_test.append(values[col])
                else:
                    col_name, col_value = col.split("_")
                    x_test.append(values[col_name] == col_value)

            print("Evalando", model_name, model_type)
            prediction_arr = model.predict([x_test])

            prediction = (
                prediction_arr[0]
                if np.isscalar(prediction_arr[0])
                else prediction_arr[0][0]
            )

            result[model_name][model_type] = (
                prediction
                if model_type == "regresion"
                else ":rain_cloud:"
                if prediction > 0.5
                else ":sun_with_face:"
            )
    return result


with st.container():
    st.header("Predicción Meteorológica Del Día Siguiente", divider="rainbow")
    with st.spinner("Ejecutando modelos..."):
        st.container()
        cols = st.columns(3, gap="small")
        for i, col_title in enumerate(
            ["Modelo", "RainfallTomorrow (Regesion)", "RainTomorrow? (Clasificacion))"]
        ):
            cols[i].subheader(col_title, divider="gray")

        evaluated_models = eval_models(values)
        for i, model_name in enumerate(evaluated_models):
            st.container()
            cols = st.columns(len(modelos[model_name]) + 1, gap="small")
            cols[0].subheader(model_name)
            for i, v in enumerate(
                [
                    evaluated_models[model_name][mt]
                    for mt in evaluated_models[model_name]
                ]
            ):
                cols[i + 1].write(v)
