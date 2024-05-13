import pandas as pd
import streamlit as st
from joblib import load

model = load("model_ridge.pkl")


def user_input_features():
    house_area = st.sidebar.slider("面積(m2)", 0.0, 250.0, 30.0)
    year_from_built = st.sidebar.slider("築年数(年)", 0, 60, 10)
    distance = st.sidebar.slider("駅からの距離(m)", 1, 7000, 160)
    balcony_area = st.sidebar.slider("バルコニーの面積(m2)", 0.0, 40.0, 10.0)
    floor = st.sidebar.slider("部屋の階数(階)", 1, 37, 1)
    total_floor = st.sidebar.slider("建物の総階数(階)", 1, 47, 3)

    data = {
        "house_area": [house_area],
        "year_from_built": [year_from_built],
        "distance": [distance],
        "balcony_area": [balcony_area],
        "floor": [floor],
        "total_floor": [total_floor],
    }

    features = pd.DataFrame(data)
    return features


st.write("# 不動産賃料推計アプリ")

for _ in range(5):  # 5行の空白を追加
    st.text("\n")


input_df = user_input_features()

price_pred = model.predict(input_df)


st.write(f"## 使用アルゴリズム : Ridge regression ")

st.write(f"## 予測結果 : {int(price_pred[0])} 円")


for _ in range(25):  # 5行の空白を追加
    st.text("\n")


# st.text("※左のスライダーで選んだ数値を元に予測します。")
# st.text("※本サイトは Streamlit を使用して作成されております。")
# st.text("表示に関する細かい設定はできませんが、簡易的なダッシュボード作成には有用とされています。")
# st.text("一般的なWebサイトほどに、表示に関する細かい設定はできません。。")
