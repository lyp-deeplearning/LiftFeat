# MegaDepth-only laptop profile
python train.py \
--name LiftFeat_MD_laptop \
--use_megadepth \
--megadepth_batch_size 1 \
--n_steps 80000 \
--lr 1e-4 \
--gamma_steplr 0.7 \
--device_num 0 \
--save_ckpt_every 2000
