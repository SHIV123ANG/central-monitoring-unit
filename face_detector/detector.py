import os
import argparse
import torch
import numpy as np
from .layers.functions.prior_box import PriorBox
from .utils.nms.py_cpu_nms import py_cpu_nms
from .models.retinaface import RetinaFace
from .utils.box_utils import decode, decode_landm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

args = argparse.Namespace()
args.trained_model = os.path.join("face_detector", "weights", "mobilenet0.25_Final.pth")
args.network = "mobile0.25"
args.cpu = (not torch.cuda.is_available())
args.confidence_threshold = 0.02
args.top_k = 5000
args.nms_threshold = 0.4
args.keep_top_k = 750
args.s = False
args.vis_thres = 0.6
args.save_image = True

net = RetinaFace(cfg=cfg, phase = 'test')
net.to(device)
state_dict = torch.load(args.trained_model, map_location=device)
net.load_state_dict(state_dict, strict=True)
del state_dict
net.eval()
resize = 1

def detect_faces(img):
    with torch.no_grad():
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        loc, conf, landms = net(img)  # forward pass
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([
            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
            img.shape[3], img.shape[2]]
        )
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()
        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]
        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]
        dets = np.concatenate((dets, landms), axis=1)
        result = []
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            b = list(map(int, b))
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            w, h = (x2-x1), (y2-y1)
            if w < 56 or h < 56:
                continue
            cx, cy = x1+(w//2), y1+(h//2)
            w, h = max(w, h), max(w, h)
            w, h = int(1.2*w), int(1.2*h)
            x1, y1 = max(0, cx-(w//2)), max(0, cy-(h//2))
            x2, y2 = min(im_width, cx+(w//2)), min(im_height, cy+(h//2))
            result.append((x1, y1, x2, y2))
    return result
