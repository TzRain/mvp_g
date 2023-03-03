cam_seq=('CMU1' 'CMU2' 'CMU3' 'CMU4' 'CMU0ex' 'CMU0ex')
cam_len=(7 7 4 10 6 7)
ck_name=('mean7' 'mean7' 'mean4' 'mean10' 'add1' 'add2')
seq_len=6

for((i=0;i<$seq_len;i++)); do
    python -m torch.distributed.launch --nproc_per_node=4 --use_env run/validate_3d.py --cfg configs/panoptic/std_CMU0.yaml GPUS=0,1,2,3 \
    DATASET.TRAIN_CAM_SEQ=${cam_seq[i]} DATASET.TEST_CAM_SEQ=${cam_seq[i]} DATASET.CAMERA_NUM=${cam_len[i]} \
    TEST.MODEL_FILE='../old/'${ck_name[i]}'.pth.tar' \
    DEBUG.WANDB_NAME=${cam_seq[i]}
done


# python -m torch.distributed.launch --nproc_per_node=4 --use_env run/validate_3d.py --cfg configs/campus/mvp_campus.yaml --model_path models/mvp_campus.pth.tar

# python -m torch.distributed.launch --nproc_per_node=4 --use_env run/validate_3d.py --cfg configs/shelf/mvp_shelf.yaml --model_path models/mvp_shelf.pth.tar