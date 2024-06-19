CUDA_VISIBLE_DEVICES=0 python -m manydepth.train \
    --data_path /home/cvlab08/projects/data/KITTI/ \
    --log_dir /home/cvlab08/projects/data/hg_log/manydepth \
    --model_name croco_disable_multiloss_withmultipose_re \
    --model_type croco \
    --croco_pretrain_path /home/cvlab08/projects/hg/manydepth/CroCo_V2_ViTBase_BaseDecoder.pth \
    --batch_size 6 \
    --disable_multiloss \
    --multipose_train \
    --freeze_teacher_epoch 21
