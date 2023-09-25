# MOT_yolov8

### Setup
```
pip install -r requirements.txt
pip install cython_bbox
```

### Download pretrained model
1. ReID model:
```
gdown --id 1N16RJ_hZGgXg3Ls5DObs3HQUKNbUSBzy
mkdir ./working/mot/weights/
cp ./working/epoch=5-Val_mAP=0.63159-Val_CMC@rank1=0.96626-Val_CMC@rank5=0.98858.ckpt ./working/mot/weights/vit_base_patch16_224_TransReID.ckpt
```

2. YOLOv8
```
gdown --id 1_oda78B4ZxJ5-LLp6X0fHqxpcs1U6feV
mv ./working/best_8.0.20.pt ./working/mot/weights/
```

### Run code
```
python track.py --source ./input/mot17-data/MOT17/MOT17/train 
--reid-weights ./working/mot/weights/vit_base_patch16_224_TransReID.ckpt
--yolo-weights ./working/mot/weights/best_8.0.20.pt
--class 0 
--tracking-method strongsort 
--save-txt
```

Output Video Save at "runs\tracks" dir