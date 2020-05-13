from keras.optimizers import Adam
import cv2
import numpy as np
from matplotlib import pyplot as plt
from losses.keras_ssd_loss import SSDLoss
from models.keras_mobilenet_v2_ssdlite import mobilenet_v2_ssd
from PIL import ImageFont, ImageDraw, Image
import colorsys
from timeit import default_timer as timer

# model config
batch_size = 32
image_size = (300, 300, 3)
n_classes = 80
mode = 'inference'
l2_regularization = 0.0005
min_scale = 0.1  # None
max_scale = 0.9  # None
scales = None  # [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
aspect_ratios_global = None
aspect_ratios_per_layer = [[1.0, 2.0, 0.5], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0], [1.0, 2.0, 0.5], [1.0, 2.0, 0.5]]
two_boxes_for_ar1 = True
steps = None  # [8, 16, 32, 64, 100, 300]
offsets = None  # [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
clip_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
coords = 'centroids'
normalize_coords = True
subtract_mean = [123, 117, 104]
divide_by_stddev = 128
swap_channels = None
confidence_thresh = 0.01
iou_threshold = 0.45
top_k = 200
nms_max_output_size = 400
return_predictor_sizes = False

model = mobilenet_v2_ssd(image_size, n_classes, mode, l2_regularization, min_scale, max_scale, scales,
                         aspect_ratios_global, aspect_ratios_per_layer, two_boxes_for_ar1, steps,
                         offsets, clip_boxes, variances, coords, normalize_coords, subtract_mean,
                         divide_by_stddev, swap_channels, confidence_thresh, iou_threshold, top_k,
                         nms_max_output_size, return_predictor_sizes)

# 2: Load the trained weights into the model.
weights_path = '../pretrained_weights/ssdlite_coco_loss-4.8205_val_loss-4.1873.h5'
model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

hsv_tuples = [(x / 81, 1., 1.) for x in range(81)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
classes = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
           'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
           'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
           'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
           'kite', 'baseball bat', 'baseball glove', 'skateboard',
           'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
           'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
           'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
           'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
           'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
           'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
           'teddy bear', 'hair drier', 'toothbrush']


def detect_image(image):
    image_resize = image.resize((image_size[:-1]))
    y_pred = model.predict(np.expand_dims(image_resize, 0))
    confidence_threshold = 0.4
    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

    font = ImageFont.truetype(font='../FiraMono-Medium.otf', size=np.floor(3e-2 * image.size[0] + 0.5).astype('int32'))

    thickness = (image.size[0] + image.size[1]) // 300
    draw = ImageDraw.Draw(image)
    for box in y_pred_thresh[0]:
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = int(box[2] * image.size[0] / image_size[0])
        ymin = int(box[3] * image.size[1] / image_size[1])
        xmax = int(box[4] * image.size[0] / image_size[1])
        ymax = int(box[5] * image.size[1] / image_size[0])
        label_size = draw.textsize(label, font)
        if xmin - label_size[1] >= 0:
            text_origin = np.array([xmin, ymin - label_size[1]])
        else:
            text_origin = np.array([xmin, ymin + 1])
        for i in range(thickness):
            draw.rectangle([xmin + i, ymin + i, xmax - i, ymax - i], outline=color)
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw
    return np.asarray(image)


def detect_video(inpath, outpath):
    capture = cv2.VideoCapture(inpath)
    video_FourCC = cv2.VideoWriter_fourcc(*'XVID')
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    video_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(outpath, video_FourCC, video_fps, video_size)

    while cv2.waitKey(1) < 0:
        t1 = timer()
        ref, frame = capture.read()
        try:
            image = Image.fromarray(frame)
        except:
            break
        image = detect_image(image)
        t2 = timer()
        fps = t2 - t1
        cv2.putText(image, 'FPS:{:.1f}'.format(1 / fps), org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", image)
        out.write(image)
    capture.release()


if __name__ == '__main__':
    detect_video('../sp.avi', '../sp_p.avi')
