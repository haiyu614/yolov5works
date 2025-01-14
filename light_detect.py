import argparse
import platform
from pathlib import Path
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (
    Profile,
    check_img_size,
    check_imshow,
    check_requirements,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
)
from utils.torch_utils import select_device
from ultralytics.utils.plotting import Annotator, colors, save_one_box


def load_model(weights, device, dnn, data, half):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((640, 640), s=stride)
    return model, stride, names, pt, imgsz


def setup_dataloader(source, imgsz, stride, pt, vid_stride):
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    return dataset


def preprocess_image(im, model):
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]
    return im


def perform_inference(im, model, augment, visualize):
    visualize = increment_path(Path("temp") / Path("temp").stem, mkdir=True) if visualize else False
    pred = model(im, augment=augment, visualize=visualize)
    return pred


def apply_nms(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det):
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    return pred


def process_predictions(im, det, im0, names, hide_labels, hide_conf, line_thickness, save_crop, save_dir, p):
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
    imc = im0.copy() if save_crop else im0
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
    if len(det):
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = names[c] if hide_conf else f"{names[c]}"
            if save_crop:
                save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)
            c = int(cls)
            label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
            annotator.box_label(xyxy, label, color=colors(c, True))
    im0 = annotator.result()
    return im0


@torch.no_grad()
def run_webcam_detection(
        weights=Path("yolov5s.pt"),
        source='0',
        data=Path("data/coco128.yaml"),
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_crop=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        project=Path("runs/detect"),
        name="exp",
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        vid_stride=1
):
    source = str(source)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "crops" if save_crop else save_dir).mkdir(parents=True, exist_ok=True)

    model, stride, names, pt, imgsz = load_model(weights, device, dnn, data, half)
    dataset = setup_dataloader(source, imgsz, stride, pt, vid_stride)
    view_img = check_imshow(warn=True)
    vid_path, vid_writer = [None] * len(dataset), [None] * len(dataset)

    model.warmup(imgsz=(1 if pt or model.triton else len(dataset), 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = preprocess_image(im, model)

        with dt[1]:
            pred = perform_inference(im, model, augment, visualize)

        with dt[2]:
            pred = apply_nms(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det)

        for i, det in enumerate(pred):
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            p = Path(p)
            im0 = process_predictions(im, det, im0, names, hide_labels, hide_conf, line_thickness, save_crop, save_dir, p)

            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=Path("yolov5s.pt"), help='model path or triton URL')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=Path("data/coco128.yaml"), help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(Path("requirements.txt"), exclude=("tensorboard", "thop"))
    run_webcam_detection(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)