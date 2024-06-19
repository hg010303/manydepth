CUDA_VISIBLE_DEVICES=3 \
python -m manydepth.visual_attn \
    --data_path /home/cvlab08/projects/data/KITTI/ \
    --load_weights_folder /home/cvlab08/projects/data/hg_log/manydepth/croco_disable_multiloss_re/models/weights_15 \
    --croco_pretrain_path /home/cvlab08/projects/hg/manydepth/CroCo_V2_ViTBase_BaseDecoder.pth \
    --model_type croco \
    --eval_mono