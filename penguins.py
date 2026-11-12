# %%
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle 

# %%
penguins = sn.load_dataset("penguins")
penguins.head()

# %%
penguins.info()
# %%
penguins.isna().sum()
# %%
penguins_df =penguins.dropna()
penguins_df.drop_duplicates()
# %%
penguins_df.info()

# %%
#penguins_df["species"].unique()
#penguins_df["species"].replace(['Adelie', 'Chinstrap', 'Gentoo'],
#                               [1,2,3],inplace=True)

teste = penguins_df["species"].astype(str)


# %%
teste.head()
teste.info()
# %%
CATEGORICAS = ["island","sex"]
NUMERICA = ["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]

x = penguins_df
y = x.pop("species")

x_treino, x_teste, y_treino, y_teste = train_test_split(x,
                                                        y,
                                                        test_size = 0.2,
                                                        random_state= 42,
                                                        stratify = y
                                                        )
# %%
y_treino.head()
# %%
pipe_cat = make_pipeline(
    OneHotEncoder(sparse_output=False, handle_unknown="ignore")
)

pipe_num = make_pipeline(
    StandardScaler()
)

preprocesador = ColumnTransformer(
    transformers=[
        ("pipe_num", pipe_num, NUMERICA),
        ("pipe_cat", pipe_cat, CATEGORICAS)
    ]
)

# %%
modelo = make_pipeline(
    preprocesador,
    RandomForestClassifier()
).set_output(transform='pandas')

modelo.fit(x_treino,y_treino)

# %%
with open('modelo_class.pkl','wb') as f:
    pickle.dump(modelo, f)

# %%
print(modelo.predict(x_teste))
# %%
