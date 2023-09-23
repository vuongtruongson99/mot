import argparse
import cv2
import os
from tqdm import tqdm
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

from tracker.bytetrack.byte_tracker import BYTETracker
from tracker.strongsort.utils.parser import get_config

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# if str(ROOT / 'yolov8') not in sys.path:
#     sys.path.append(str(ROOT / 'yolov8'))  # add yolov8 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH


ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from ultralytics.nn.autobackend import AutoBackend  # For inference on various platform
from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from tracker.get_trackers import create_tracker

def run(opt):
    source = opt.seq                         # youtube, video,...
    yolo_weights = opt.yolo_weights              # yolov8l.pt
    reid_weights = opt.reid_weights             # osnet_x0_25_msmt17.pt
    tracking_method = opt.tracking_method       # bytetrack
    tracking_config = opt.tracking_config       # bytetrack.yaml
    imgsz = opt.imgsz                           # inference size: 640x640
    conf_thres = opt.conf_thres                # threshold for detector
    iou_thres = opt.iou_thres                   # For first stage association
    max_det = opt.max_det                       # Maximum detections per img
    device = opt.device
    show_vid = opt.show_vid                     # Display tracking video results
    save_txt = opt.save_txt                     # save results to *.txt
    save_conf = opt.save_conf                   # save confidence to --save_txt file
    save_crop = opt.save_crop                   # save cropped prediction boxes (for reid training)
    save_trajectories = opt.save_trajectories
    save_vid = opt.save_vid
    nosave =opt.nosave
    classes = opt.classes                       # Only track this classes
    agnostic_nms = opt.agnostic_nms
    augment = opt.augment
    visualize = opt.visualize
    update = opt.update                         # update all models
    project = opt.project                       # save results to project/name
    name = opt.name                             # save results to project/name
    exist_ok = opt.exist_ok                     # existing project/name ok, do not increment
    line_thickness = opt.line_thickness         # bounding box thickness (pixels)
    hide_labels = opt.hide_labels               # hide labels
    hide_conf = opt.hide_conf                   # hide confidences
    hide_class = opt.hide_class                 # hide IDs
    half = opt.half                             # use FP16 half-precision inference
    dnn = opt.dnn                               # use OpenCV DNN for ONNX inference
    vid_stride = opt.vid_stride                 # video frame-rate stride
    retina_masks = opt.retina_masks

    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # print(save_dir, exp_name)

    # Load model
    device = select_device(device)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size
    if half:
        model.half()    # to FP16

    # Dataloader
    bs = 1
    dataset = LoadImages(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # print(dataset)

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * bs

    # cfg = get_config()
    # cfg.merge_from_file(tracking_config)
    # tracker = BYTETracker(cfg)
    # Run tracking
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs


    for frame_idx, batch in tqdm(enumerate(dataset)):
        path, im, im0s, vid_cap, s = batch
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False

        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()
            im /= 255.0
            if len(im.shape) == 3:
                im = im[None]   # expand for batch dim

        # Inference
        with dt[1]:
            preds = model(im, augment=augment, visualize=visualize)
            # print(len(preds[0]), len(preds[0][0]), len(preds[0][0][0]))
        with dt[2]:
            p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # print(f"{frame_idx}: {len(p[0])}")
        # Process detections
        for i, det in enumerate(p):
            seen += 1
            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)
            txt_file_name = p.parent.parent.name
            save_path = str(save_dir / p.parent.parent.name)    # runs/track/yolov8l/MOT20-01
            curr_frames[i] = im0
            txt_path = str(save_dir/ 'tracks' / txt_file_name)  # runs/track/yolov8l/tracks/MOT20-01
            s += '%gx%g ' % im.shape[2:]
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)

                if len(outputs[i]) > 0:
                    for j, (output) in enumerate(outputs[i]):
                        
                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]
                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                            (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                        color = colors(id % 100, True)
                        annotator.box_label(bbox, label, color=color)

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, id, int(bbox_left),  # MOT format
                                                               int(bbox_top), int(bbox_w), int(bbox_h), -1, -1, -1, -1))
                # print("After update: ", online_target)
                # for t in online_target:
                #     tlwh = t.tlwh
                #     tlwh[2:] += tlwh[:2]
                #     id = t.track_id
                #     color = colors(id, True)
                #     label = None if hide_labels else (f'{id}')
                #     annotator.box_label(tlwh, label, color=color)

                
                
                
                # if len(outputs[0]) > 0:
                #     for j, (output) in enumerate(outputs[0]):
                        
                #         bbox = output[0:4]
                #         id = output[4]
                #         cls = output[5]
                #         conf = output[6]

                #         c = int(cls)  # integer class
                #         id = int(id)  # integer id
                #         label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                #             (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                #         color = colors(c, True)
                #         annotator.box_label(bbox, label, color=color)

            
            im0s = annotator.result()
            if save_vid:
                save_path = str(Path(save_path))  # force *.mp4 suffix on results videos
                os.makedirs(save_path, exist_ok=True)
                img_name = path.split("/")[-1]
                out_img_path = os.path.join(save_path, img_name)
                cv2.imwrite(out_img_path, im0)


            # cv2.imshow("Test", im0s)
            # if cv2.waitKey(1) == ord('q'):  # 1 millisecond
            #     exit()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov8x.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='bytetrack', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='MOT20', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', default=False, action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')

    # ByteTrack arguments
    parser.add_argument("--track_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value."
                        )
    parser.add_argument("--use_bbox_filters", default=False, action="store_true",
                        help="use ByteTrack bbox size and dimensions ratio filters")

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'tracker' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    # print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    for seq in os.listdir(opt.source):
        print("[INFO] Processing", seq)
        img_path = os.path.join(opt.source, seq, 'img1')
        opt.seq = img_path
        run(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

    
