import pandas as pd
import streamlit as st
from joblib import load

model = load("model_ridge.pkl")


def user_input_features():
    house_area = st.sidebar.slider("面積(m2)", 0.0, 200.0, 30.0)
    distance = st.sidebar.slider("駅からの距離(m)", 1, 2000, 160)

    data = {"house_area": [house_area], "distance": [distance]}

    features = pd.DataFrame(data)
    return features


st.write("# 不動産価格予測アプリ")

input_df = user_input_features()

price_pred = model.predict(input_df)

st.write(f"## 予測結果: {int(price_pred[0])} (円)")
