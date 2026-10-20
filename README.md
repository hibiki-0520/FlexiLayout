# FlexiLayout

倉庫レイアウト自動設計・ピッキングシミュレーションシステムです。

棚・作業者・コンテナ・フォークリフトの数を入力し、レイアウト案の設計と
SimPy による離散事象シミュレーションを行うためのプロジェクト構成です。
シミュレーション結果から作業者の動線や設備の待ち時間などを評価し、
グラフとして可視化します。

## ディレクトリ構成

```text
.
├── data/
│   ├── input/                 # 入力データ、シナリオ設定
│   └── output/                # シミュレーション結果、出力画像
├── docs/                      # 設計資料
├── src/
│   └── warehouse_sim/
│       ├── layout/            # 棚・通路などのレイアウト生成、最適化
│       ├── simulation/        # SimPyモデル、動線・ボトルネック評価
│       └── visualization/     # レイアウト図、シミュレーション結果の可視化
├── tests/                     # 自動テスト
├── README.md
└── requirements.txt
```

## 主な依存ライブラリ

- **SimPy**: 作業、搬送、資源待ちを離散事象シミュレーションで表現
- **SciPy**: レイアウト候補の最適化
- **NetworkX**: 通路ネットワークと動線の分析
- **NumPy / pandas**: 計算、シミュレーション結果の集計
- **Matplotlib / Seaborn**: レイアウト、稼働状況、待ち時間などの可視化

## グリッド生成と棚割り

`WarehouseGrid` は2次元グリッドに通路とI/Oポイントを配置し、商品を棚セルへ
割り当てます。ABC区分列がなければ、ピッキング頻度の累積比率から A/B/C
（80%/15%/5%）を算出し、A商品からI/Oポイントに近い棚へ配置します。

```python
import pandas as pd

from warehouse_sim.layout.grid import WarehouseGrid
from warehouse_sim.visualization import plot_warehouse_3d, plot_warehouse_grid

warehouse = WarehouseGrid(width=50, height=50, aisle_spacing=8)
products = pd.DataFrame(
    {
        "product_id": ["item-a", "item-b", "item-c"],
        "pick_frequency": [800, 150, 50],
    }
)
slotting = warehouse.optimize_slotting(products)
print(slotting[["product_id", "abc_class", "row", "column"]])

figure = plot_warehouse_grid(warehouse.grid, slotting)
figure_3d = plot_warehouse_3d(warehouse.grid, slotting)
```

I/Oポイントは `(row, column)` のリストで指定でき、省略時は下辺の中央に1点
配置されます。`warehouse.grid` は `.`（床）、`A`（通路）、`I`（出入口）、
`S`（棚）を表すNumPy配列です。`plot_warehouse_grid` はセル種別ごとの凡例を
表示し、棚割りDataFrameを渡すと棚をABC区分ごとに色分けします。
`plot_warehouse_3d` はPlotlyで棚を高さのある直方体として描画し、I/Oポイント、
通路、ABC別の棚を3D表示します。`worker_paths` と `forklift_paths` を渡すと
それぞれの軌跡も表示できます。Streamlitアプリでは2D/3Dタブを切り替えられ、
3D図は暗色グリッド床、通路帯、複数段の棚板とスチールフレームからなるラック、
ABC区分別の棚板、立体的なI/Oドックを表示します。ドラッグで回転、スクロールで
ズームできます。

## ピッキングシミュレーション

`simulate_picking` はSimPyでオーダー処理を実行し、完了時間（メイクスパン）、
作業者・フォークリフトの稼働率と待ち時間を計算します。商品ごとの棚座標は
先ほどの棚割りDataFrameから作成できます。シミュレーション時間の単位は任意で、
速度はグリッドセル/時間単位、ピッキング時間は個/時間単位です。

```python
from warehouse_sim.simulation import PickTask, PickingOrder, simulate_picking

orders = [
    PickingOrder("order-1", [PickTask("item-a", 2), PickTask("item-b")]),
    PickingOrder("order-2", [PickTask("item-c")]),
]
result = simulate_picking(
    orders,
    shelf_positions={"item-a": (10, 8), "item-b": (12, 8), "item-c": (15, 9)},
    io_point=(49, 25),
    worker_count=3,
    forklift_count=1,
    container_count=2,
    worker_speed=1.0,
    forklift_speed=2.0,
    pick_time_per_unit=0.5,
)
print(f"Makespan: {result.makespan:.2f}")
print(f"Worker utilization: {result.worker_utilization:.1%}")
print(f"Forklift utilization: {result.forklift_utilization:.1%}")
print(f"Worker wait: {result.worker_wait_time_total:.2f}")
print(f"Forklift wait: {result.forklift_wait_time_total:.2f}")
```

同時処理するオーダーは、作業者・コンテナ・フォークリフトのSimPyリソースを
要求します。Streamlitアプリは倉庫グリッド上でA*探索を行い、棚セルを避ける
最短の上下左右経路を移動します。移動時間はその経路のグリッド辺数と速度から
計算し、各辺を記録間隔ごとに補間して位置を記録します。オーダーごとに作業者は
I/Oポイントから出発します。作業者・フォークリフトの稼働率は
ピッキング/移動または運搬の実作業時間を、メイクスパン中の全リソース時間で
割った値です。各リソースの待ち時間は合計と1要求あたり平均を返します。
アプリでは荷物重量、サイズ区分、1回あたりの搬送上限を指定できます。重量と
サイズに応じて作業速度・ピッキング時間・積載時間を調整し、搬送上限を超える
数量は複数バッチに分けます。小型荷物は作業員が手持ちで戻し、中型はコンテナと
フォークリフト、大型はフォークリフトを使用します。

シミュレーション結果には、時間ごとのエージェント位置・状態を記録した
`result.trace`（pandas DataFrame）が含まれます。列は `time`, `agent_id`,
`agent_type`, `x`, `y`, `state`, `wait_time_total`,
`queue_count`, `completed_order_count` です。`simulate_picking` は必要に応じて位置ログを記録できます。Streamlitアプリでは
パフォーマンスを優先して位置ログとブラウザー内アニメーションを省略し、メイクスパン、
稼働率、待ち時間などのKPIと2Dレイアウトを表示します。インタラクティブな3D倉庫と
AGVアニメーションは、別ページの `index.html` で確認できます。

`optimize_conditions` は同じ注文データで作業員数、フォークリフト数、
棚配置ルール（ABC優先/頻度優先）を比較します。時間優先、予算内コスト最小化、
最低稼働率を考慮した費用対効果、フォークリフトの走行距離・稼働時間に基づく
CO2換算推定値の4目的を選べます。候補KPIと現在条件との比較表、推奨理由を表示し、
推奨条件をアプリの入力欄へ適用できます。費用とCO2換算値は明示された係数を用いる
モデル推定値であり、実測値や実費用ではありません。

## セットアップ

```bash
python -m pip install -r requirements.txt
```

## Streamlitアプリ

棚数、作業員数、フォークリフト数をブラウザ上で変更し、レイアウトと
シミュレーションKPIを確認できます。コンテナ数は作業員数と同数に設定します。

```bash
python -m streamlit run app.py
```

Streamlitアプリは `http://127.0.0.1:8765/index.html` へのリンクを表示します。
3D Studioを使う場合は、プロジェクトのルートで別のターミナルを開いて次も実行してください。

```bash
python -m http.server 8765
```

システムの機能、計算方式、前提・制約については
[システム仕様書](./docs/システム仕様書.md) を参照してください。
