# 不動産賃料推計アプリ

物件の条件を入力すると、機械学習モデルで賃料を推計して表示するStreamlitアプリです。

## 主な機能

- 住所、市区町村、駅、間取りを選択
- 専有面積、築年数、所在階、建物の総階数、駅からの徒歩時間を指定
- 入力条件に基づく推計結果を円単位で表示

## 動作環境

- Python 3.11（Dev Containerの設定に準拠）
- pip

## セットアップと起動

リポジトリのルートディレクトリで、次のコマンドを実行します。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run streamlit_predict.py
```

Windowsでは仮想環境を次のコマンドで有効にしてください。

```powershell
.venv\Scripts\Activate.ps1
```

起動後、ターミナルに表示されるURL（通常は http://localhost:8501 ）をブラウザーで開きます。

## リポジトリ内の主なファイル

| ファイル | 説明 |
| --- | --- |
| `streamlit_predict.py` | 入力フォームと賃料推計結果を表示するStreamlitアプリ |
| `model_LightGBM.pkl` | アプリが推計に使用するLightGBMモデル |
| `model_ridge.pkl` | 同梱されているRidgeモデル |
| `summo_Pretreated_20240521.csv` | 入力候補と項目範囲の読み込みに使う前処理済みデータ |
| `requirements.txt` | Pythonパッケージ一覧 |

モデルとCSVファイルはアプリ起動時に読み込まれるため、リポジトリのルートに配置してください。

## 推計結果について

表示される金額はモデルによる推計値です。実際の募集賃料や契約条件を保証するものではありません。
