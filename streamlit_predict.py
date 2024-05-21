import pandas as pd
import streamlit as st
from joblib import load

# モデルのロード
model = load("model_LightGBM.pkl")

# データの読み込み
df = pd.read_csv("./summo_Pretreated_20240521.csv")

# カテゴリ変数の列名をリストで指定
categorical_features = ["住所", "間取り", "市区町村", "駅"]


# ユーザー入力を受け取る関数
def user_input_features():

    # カテゴリカル変数の選択ボックス
    address = st.sidebar.selectbox("住所", df["住所"].unique())
    city = st.sidebar.selectbox("市区町村", df["市区町村"].unique())
    # sub_city = st.sidebar.selectbox("市区町村以下", df["市区町村以下"].unique())
    # line = st.sidebar.selectbox("路線", df["路線"].unique())
    station = st.sidebar.selectbox("駅", df["駅"].unique())
    layout = st.sidebar.selectbox("間取り", df["間取り"].unique())

    # スライダーの設定
    area = st.sidebar.slider(
        "専有面積(m2)", float(df["専有面積"].min()), float(df["専有面積"].max()), float(df["専有面積"].quantile(0.25))
    )

    year_from_built = st.sidebar.slider(
        "築年数", int(df["築年数"].min()), int(df["築年数"].max()), int(df["築年数"].quantile(0.25))
    )

    floor = st.sidebar.slider(
        "階",
        int(df["階"].min()),
        int(df["階"].max()),
        int(df["階"].quantile(0.25)),
    )

    total_floor = st.sidebar.slider(
        "建物の総階数",
        int(df["総階数"].min()),
        int(df["総階数"].max()),
        int(df["総階数"].quantile(0.25)),
    )

    distance = st.sidebar.slider(
        "駅から徒歩 (分)",
        int(df["歩"].min()),
        # int(df["歩"].max()),
        int("20"),
        int(df["歩"].quantile(0.25)),
    )

    data = {
        "住所": [address],
        "築年数": [year_from_built],
        "総階数": [total_floor],
        "階": [floor],
        # "管理費": [management_fee],
        # "地下": [basement],
        # "地上": [above_ground],
        "間取り": [layout],
        "専有面積": [area],
        "市区町村": [city],
        # "市区町村以下": [sub_city],
        # "路線": [line],
        "駅": [station],
        "歩": [distance],
    }

    features = pd.DataFrame(data)

    # カテゴリカル特徴量をカテゴリ型に変換
    for col in categorical_features:
        features[col] = features[col].astype("category")

    return features


st.write("# 不動産賃料推計アプリ")

# 空白行を追加
for _ in range(1):
    st.text("\n")

# ユーザー入力を受け取る
input_df = user_input_features()

# 予測
price_pred = model.predict(input_df)

# 予測結果を表示
st.write(f"## 予測結果 : {int(price_pred[0] * 10000)} 円")

# 空白行を追加
for _ in range(25):
    st.text("\n")

st.text("※左のスライダーで選んだ数値を元に予測します。")
st.text("※本サイトは Streamlit を使用して作成されております。")
st.text("表示に関する細かい設定はできませんが、簡易的なダッシュボード作成には有用とされています。")
st.text("一般的なWebサイトほどに、表示に関する細かい設定はできません。")
