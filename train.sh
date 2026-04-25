HF_ENDPOINT=https://hf-mirror.com
python -m tools.train_all_classes \
  --parallel false \
  --visible_devices 0 \
  --classes 0 1 2 3 \
  --data_dir ./data/ToolWear_RGB \
  --save_root ./checkpoints_4cls \
  --sd_path ./sd15 \
  --epoch 100 \
  --batch_size 32 \
  --lr 1e-4 \
  --step_number 8 \
  --cs_ratio 0.1 \
  --block_size 8