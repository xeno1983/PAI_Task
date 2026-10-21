# Colab での実行（任意）

提出手法はローカルと同じく **教師 CPG → BC → PPO** です。

1. このフォルダを zip して Colab にアップロードし展開する。
2. GPU ランタイムを選ぶ。
3. 依存関係を入れたあと、例えば次を実行する。

```python
!python -m pick_and_carry.train_bc --easy-walk --num-envs 8 --seconds 30 --epochs 50 --log-dir logs/imitation --arm z1
!python -m pick_and_carry.train_ppo --easy-walk --init-from logs/imitation/bc_trot.pt --num-envs 8 --iterations 50 --curriculum walk --arm z1 --log-dir logs/ppo_curriculum_walk_easy
```

詳細な引数はリポジトリ直下の `README.md` を参照してください。
