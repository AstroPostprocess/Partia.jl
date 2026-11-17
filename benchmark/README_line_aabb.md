# Line–AABB / LBVH benchmarks (CPU, D = 3)

從 repository 根目錄執行，不需要額外安裝 BenchmarkTools 或 ArgParse：

```powershell
julia --project=. benchmark/benchmark_line_aabb.jl --n 100000 --repeats 10 --type Float64
julia --project=. benchmark/benchmark_line_aabb.jl --n 100000 --repeats 10 --type Float32
julia --project=. benchmark/benchmark_line_traversal.jl --nparts 10000,100000,1000000 --nlines 1000 --repeats 10 --type Float64
julia --project=. benchmark/benchmark_line_traversal.jl --nparts 10000,100000 --nlines 1000 --repeats 10 --type Float32 --radius 0.02
```

輸出為帶有 `#` metadata 的 CSV，可用 `> results.csv` 保存。兩支皆接受 `--seed 20261118`。
若 Windows Julia precompilation 出現 `pipe_open: bad file descriptor`，可加上
`--compiled-modules=no`（放在 script 路徑之前）；每個測量仍先 warm-up。

## 比較項目

* `lower_bound`：目前 production 的 `_squared_distance_line_aabb_lower_bound`。
* `exact_new`：工作目錄中新增的 `_squared_distance_line_aabb`，直接呼叫，不另複製。
* `eberly_3d`：Eberly [DistLine3CanonicalBox3.h](https://github.com/davideberly/GeometricTools/blob/master/GTE/Mathematics/DistLine3CanonicalBox3.h)
  的 Julia squared-distance-only 移植，保留 3D face/edge/corner 與零方向分量分支。
  重複 edge 分支抽成 helper，省略 closest points、parameter 輸出及最後 sqrt，
  結果以 `max(d2,0)` 修正相減造成的負浮點誤差。不是 C++ 原碼的速度比較。
  授權與來源見 `eberly_line_aabb.jl`、`LICENSE_EBERLY.txt`。

使用無限直線，所有方向正規化；每種方法都包含 AABB center/extent 轉換成本。
舊方法只是 lower bound，不能要求一般 box 上三者距離相等。

## 單次距離

實際計時是一整批獨立 line–box 查詢，輸出 min／median ns/query、整批 allocated bytes、
checksum，避免單次呼叫的時鐘成本主導結果。包含 random／axis／plane 方向，搭配一般盒、
薄盒、退化點盒、確定相交的直線。計時前，以獨立 Float64 piecewise-quadratic reference
驗證所有輸入：新方法與 Eberly 接近精確值，舊方法不得超過精確值（允許浮點誤差）。
距離誤差界限為 `128 * eps(T) * max(1, reference_d2, geometry_scale2)`；
geometry_scale2 是原點至 AABB 最遠角點的平方距離，反映平方量相減的捨入尺度。

## LBVH traversal

使用 Partia 的 Morton sort 與 LinearBVH 建樹，建樹不計時。三者共用 benchmark 內同一個
stackless DFS，遵循 production macro 的 left／escape、leaf acceptance 與 radius 規則，
只替換距離函式；沒有改 production macro。callback 為累加命中數，不包含 kernel interpolation。
測量為單執行緒 CPU traversal；不是 GPU 或完整 interpolation benchmark。

uniform／clustered 資料使用 point leaves（符合目前粒子 interpolation 用法）。
gather 半徑固定為 `--radius`；scatter 使用每粒子 `[0.5,1.5) * radius` 的 scale，Kvalid=1。
內部節點使用建樹所產生的最大 scale。輸出 min／median us/line、整批 allocated bytes、
整批命中數及平均 visited nodes／tested leaves。這能區分「距離計算較貴」與「剪枝較好」。

`--validate 32` 控制每組前幾條直線的 brute-force leaf-ID 集合驗證與統計（預設 32）。
同時對照目前 production gather／scatter macro 回傳的 leaf-ID 集合。
節點統計另跑 instrumented traversal，不加入計時；整批三者命中數也必須相等。
若要驗證／統計所有直線，設定 `--validate` 等於 `--nlines`。Float32 的極近邊界查詢可能因
捨入產生不同判斷；驗證會直接失敗而不忽略差異。

每組先 warm-up，再重複 `@timed` 整批查詢；GC 在每次計時前執行，計時中 GC 成本保留。
請以相同 Julia 版本、型別、資料量與 seed 比較，增加 repeats 觀察穩定性。

## Phantom 真實 gas／dust 資料

```powershell
julia --project=. benchmark/benchmark_phantom_line.jl --files temp/st15co_00000,temp/st15co_00120,temp/disc_00102 --nlines 256 --repeats 5 > benchmark/phantom_results.csv
```

需要可載入 `ParticleIO` 的 Julia environment（這台機器的 global environment 已有）。
不修改 Partia 的套件依賴。預設讀取上述三個檔案，以 `separate_types=:all` 分開有效粒子，
itype=1 為 gas，itype>=7 以 `dust_typeN` 標示，其餘以 `typeN` 標示，sinks 不進入粒子樹。
全部有效粒子建樹，不抽樣縮小 N；實際各類粒子數會列在 metadata。

以 `--sink 1` 的位置為座標原點（M1；只是平移，不裁切粒子），列出相對距離與 h 的分位數。
每個 population 以 PCA 估計盤面法向與盤內主軸，測試 random／face_on／edge_on 三種方向。
查詢原點從該 population 的粒子均勻抽樣，各座標加上 `[-h,h]` 擾動，集中測量有粒子的區域。
這是依粒子分布加權的 workload，並非均勻影像 pixel grid；gas／dust 各自抽樣，
同一組的三種演算法則共用完全相同查詢。

scatter 使用真實 h 與 `--kvalid 3`；gather 固定用 `Kvalid * median(h)`。
讀檔、PCA、Morton sort、建樹、驗證全部排除在計時外。
`stage=distance` 使用 `--nmicro 10000` 筆均勻抽樣的真實內部節點 AABB，
不代表 traversal 中節點出現的頻率；其 checksum 是距離總和，舊 lower bound 預期不同。
`stage=traversal` 測量整批直線，checksum_or_hits 為命中總數，三者必須一致。
兩種 stage 的時間單位都是 **us/query**。CSV 的 `#` 行包含設定與資料摘要。

其餘參數：`--type Float32|Float64`、`--seed 20261118`、`--validate 8`。
預設驗證每組前 8 條線的完整 leaf-ID 集合與 production macro，其餘線比較整批命中數。

### 本機試跑（2026-11-18）

`phantom_line_f64.csv` 是三份 dump 的完整有效粒子樹試跑結果：

```powershell
julia --compiled-modules=existing --project=. benchmark/benchmark_phantom_line.jl --nlines 64 --nmicro 2000 --validate 4 --repeats 3 > benchmark/phantom_line_f64.csv
```

Julia 1.12.5、CPU alderlake、Float64，所有距離檢查、選定查詢的 brute-force／production
leaf-ID 集合檢查、整批命中數比較皆通過。`--compiled-modules=existing` 是本機 Julia 1.12
避開 precompile pipe 問題的執行選項；舊 Julia 可使用前述 `--compiled-modules=no`。

| dump | gas 有效粒子 | dust 有效粒子 |
|---|---:|---:|
| st15co_00000 | 500,000 | 50,000 |
| st15co_00120 | 448,250 | 49,767 |
| disc_00102 | 453,625 | 48,784 |

例如 `st15co_00120` 的 random scatter，中位數 **us/line**：

| population | lower_bound | exact_new | eberly_3d |
|---|---:|---:|---:|
| gas | 191.53 | 287.35 | 163.53 |
| dust | 22.37 | 43.28 | 19.83 |

這次新方法的較強剪枝尚未抵銷計算成本。以上只測 64 條線、3 次重複；
正式選型請增加 nlines／repeats，並比較 CSV 裡全部方向與 gather／scatter。

Float32 的 `phantom_line_f32_smoke.csv` 是**未完成且驗證失敗**的紀錄，不能當成等價查詢的效能比較。
執行參數為 `--files temp/st15co_00120 --type Float32 --nlines 16 --nmicro 500 --validate 4 --repeats 1`。
gas 的 edge_on gather 命中總數分別為 lower_bound=81,625、exact_new=81,621、eberly_3d=81,620，
因此腳本按設計停止，沒有繼續產生 dust 結果。Float64 上述完整試跑通過。

將同一命令改成 `--validate 16` 可取得第一個 leaf mismatch 的診斷：gas、edge_on、gather，
query=5、leaf=43843。`radius2=7.820139`，Float64 reference 距離平方為 `7.827323943624261`；
lower_bound=`7.81836`（誤接受）、exact_new=`7.8273234`、eberly_3d=`7.8203125`。
這個例子顯示 Float32 平方量相減的精度損失；新方法在這個 leaf 上與 reference 接近。
benchmark 會報告方法、query、leaf、radius2 與三者距離，沒有修改 production 距離實作。
