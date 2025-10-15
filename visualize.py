import random
import cv2

def get_unique_color():
    # 使用哈希函数生成一个固定的颜色
    random.seed(7)
    color_dict = dict()
    for i in range(100):
        color_dict[i] = (random.randint(0, 150), random.randint(0, 150),
                         random.randint(0, 150))
    return color_dict

COLORS = get_unique_color()
FONTSCALE=0.8
THICKNESS=2

def visualize_func(detections, texts, image, save_path=None, use_tracker=False,
              tracked_idx=None, size=None): # size:(w,h)

    labels = []
    for i, (class_id, confidence) in enumerate(
            zip(detections.class_id, detections.confidence)):
        if detections.tracker_id is not None:
            labels.append(
                f"{texts[class_id]} {confidence:0.2f} ID:{detections.tracker_id[i]}")
        else:
            labels.append(f"{texts[class_id]} {confidence:0.2f}")

    # 读取图像
    if isinstance(image, str):
        image = cv2.imread(image)

    # 创建一个副本用于叠加半透明区域
    overlay = image.copy()

    # 绘制左上角的额外文本
    text_write = f"keyword: {','.join(texts)}"
    (text_w, text_h), baseline = cv2.getTextSize(
        text_write, cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONTSCALE,
        thickness=THICKNESS)
    x1, y1 = 10, 10  # 左上角坐标
    cv2.rectangle(
        image,
        (x1, y1),
        (x1 + text_w + 10, y1 + text_h + baseline + 10),
        color=(0, 0, 255),  # 背景框颜色
        thickness=-1
    )
    cv2.putText(
        image,
        text_write,
        (x1 + 5, y1 + text_h + baseline + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=FONTSCALE,
        color=(255, 255, 255),  # 文字颜色（白色）
        thickness=THICKNESS
    )

    track_ids = detections.tracker_id if detections.tracker_id is not None \
        else [None] * len(detections.xyxy)

    track_bboxes = detections.data["track_boxs"] if "track_boxs" in detections.data.keys() \
        else [None] * len(detections.xyxy)
    # 遍历每个检测框
    for i, (bbox, label, tid, track_box) in enumerate(
            zip(detections.xyxy, labels, track_ids, track_bboxes)):
        x1, y1, x2, y2 = map(int, bbox)

        # 确定颜色：如果是跟踪的目标或指定的索引，使用红色
        if (tracked_idx is not None and i == tracked_idx):
        # if (tracked_idx is not None and tid == tracked_idx):
            color = (0, 0, 255)  # 红色 (BGR)
            # 在overlay上绘制填充的矩形
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                color=color,
                thickness=-1  # 填充
            )
        else:
            if tid is not None:
                color = COLORS[tid % 100]
            else:
                color = (255, 0, 0)  # 蓝色 (BGR)

        # 绘制检测框
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color=color,
            thickness=2
        )

        if track_box is not None:
            tx1, ty1, tx2, ty2 = map(int, track_box)
            cv2.rectangle(
                image,
                (tx1, ty1),
                (tx2, ty2),
                color=(0,0,0),
                thickness=2
            )

        # 计算文字尺寸
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=FONTSCALE,
            thickness=THICKNESS
        )
        y_text = max(y1, text_h + baseline + 2)

        # 绘制标签背景
        cv2.rectangle(
            image,
            (x1, y_text - text_h - baseline - 2),
            (x1 + text_w + 2, y_text),
            color=color,
            thickness=-1
        )
        # 绘制文字
        cv2.putText(
            image,
            label,
            (x1 + 1, y_text - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=FONTSCALE,
            color=(255, 255, 255),
            thickness=THICKNESS
        )

    # 将半透明的overlay叠加到原图像上
    alpha = 0.5  # 透明度
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    if size is not None:
        image = cv2.resize(image, size)
    # 保存或返回
    if save_path:
        cv2.imwrite(save_path, image)
    return image
