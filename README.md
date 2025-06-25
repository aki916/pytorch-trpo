# PyTorch implementation of TRPO

Try my implementation of [PPO](github.com/ikostrikov/pytorch-a2c-ppo-acktr/) (aka newer better variant of TRPO), unless you need to you TRPO for some specific reasons.

##

This is a PyTorch implementation of ["Trust Region Policy Optimization (TRPO)"](https://arxiv.org/abs/1502.05477).

This is code mostly ported from [original implementation by John Schulman](https://github.com/joschu/modular_rl). In contrast to [another implementation of TRPO in PyTorch](https://github.com/mjacar/pytorch-trpo), this implementation uses exact Hessian-vector product instead of finite differences approximation.

## 新機能: GIF可視化

学習の進捗をGIFアニメーションで可視化する機能を追加しました。

### 使用方法

```bash
# 基本的な学習実行（GIF保存間隔: 100エピソード）
python main.py --env-name "Reacher-v4" --save-gif-interval 100

# より頻繁にGIFを保存（50エピソードごと）
python main.py --env-name "Reacher-v4" --save-gif-interval 50

# GIFの長さを調整（3秒）
python main.py --env-name "Reacher-v4" --gif-duration 3

# 出力ディレクトリを指定
python main.py --env-name "Reacher-v4" --output-dir "my_gifs"
```

### 可視化オプション

- `--save-gif-interval`: GIF保存間隔（デフォルト: 100エピソード）
- `--gif-duration`: GIFの長さ（秒）（デフォルト: 5秒）
- `--output-dir`: 出力ディレクトリ（デフォルト: "gifs"）
- `--plot-interval`: 報酬プロット更新間隔（デフォルト: 50エピソード）

### 監視ツール

学習中にGIFファイルの生成を監視するツールも提供しています：

```bash
# GIFディレクトリを監視
python visualize_training.py --mode monitor --output-dir gifs
```

## Contributions

Contributions are very welcome. If you know how to make this code better, don't hesitate to send a pull request.

## Usage

```
python main.py --env-name "Reacher-v1"
```

## Recommended hyper parameters

InvertedPendulum-v1: 5000

Reacher-v1, InvertedDoublePendulum-v1: 15000

HalfCheetah-v1, Hopper-v1, Swimmer-v1, Walker2d-v1: 25000

Ant-v1, Humanoid-v1: 50000

## Results

More or less similar to the original code. Coming soon.

## Todo

- [x] GIF可視化機能
- [x] 報酬プロット機能
- [ ] Plots.
- [ ] Collect data in multiple threads.
