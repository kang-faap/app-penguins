from flask import Flask, jsonify, request
import pickle
import pandas as pd


SPECIES = {
    1: "Adelie", 2:'Chinstrap',3:'Gentoo'
}

with open("modelo_ordinal.pkl", "rb") as arquivo:
    modelo_ordinal = pickle.load(arquivo)

with open("modelo_label.pkl", "rb") as arquivo:
    modelo_label = pickle.load(arquivo)

with open("modelo_class.pkl", "rb") as arquivo:
    modelo_class = pickle.load(arquivo)

app = Flask(__name__)

# Rota para receber dados via POST
@app.route("/predict", methods=["POST"])
def predict():
    dados = request.get_json()
    df = pd.DataFrame([dados])
    resultado_ordinal = modelo_ordinal.predict(df)
    resultado_label = modelo_label.predict(df)
    resultado_class = modelo_class.predict(df)

    ad = resultado_label[0][0]
    chi = resultado_label[0][1]
    gen = resultado_label[0][2]

    return {
        'species-class':resultado_class[0],
        'species-ordinal':SPECIES[round(resultado_ordinal[0])],
        'species-label':{
            SPECIES[1]:ad,
            SPECIES[2]:chi,
            SPECIES[3]:gen
        }
    }
    

@app.route("/predict-class", methods=["POST"])
def predict_regresor():
    dados = request.get_json()
    df = pd.DataFrame([dados])
    #resultado_ordinal = modelo_ordinal.predict(df)
    #resultado_label = modelo_label.predict(df)
    resultado_class = modelo_class.predict(df)

    return {
        'species-class':resultado_class[0]
    }

"""

curl -X POST "http://localhost:5000/predict-class" \
  -H "Content-Type: application/json" \
  -d "{
    \"island\": \"Torgersen\",
    \"bill_length_mm\": 39.1,
    \"bill_depth_mm\": 18.7,
    \"flipper_length_mm\": 181.0,
    \"body_mass_g\": 3750.0,
    \"sex\": \"Male\"
  }"

"""