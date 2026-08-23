$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

& "C:\Users\Admin\miniconda3\envs\torch\python.exe" .\train.py `
  --name LiftFeat_MD_laptop `
  --use_megadepth `
  --megadepth_root_path "E:\LiftFeat\dataset\MegaDepth\phoenix\S6\zl548" `
  --megadepth_batch_size 1 `
  --ckpt_save_path "E:\LiftFeat\trained_weights\megadepth_laptop" `
  --n_steps 80000 `
  --lr 1e-4 `
  --gamma_steplr 0.7 `
  --device_num 0 `
  --save_ckpt_every 2000
