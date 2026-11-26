#!/usr/bin/env python

import os
import csv
import argparse
import itertools
from math import degrees
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np

from utils import CvFpsCalc
from utils.utils import rotate_and_crop_rectangle
from utils.tracker import HandTracker
from model import PalmDetection
from model import HandLandmark
from model import KeyPointClassifier
from model import PointHistoryClassifier

APP_DIR = os.path.dirname(os.path.abspath(__file__))
KEYPOINT_CSV = os.path.join(APP_DIR, 'model/keypoint_classifier/keypoint.csv')
POINT_HISTORY_CSV = os.path.join(APP_DIR, 'model/point_history_classifier/point_history.csv')
POINTER_SIGN_ID = 2      # keypoint_classifier_label.csv: Open, Close, Pointer
INDEX_FINGER_TIP = 8
HISTORY_LENGTH = 16      # point_history_classifier input = HISTORY_LENGTH * 2
GESTURE_HISTORY_LENGTH = 10


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--device', type=int, default=0)
    parser.add_argument('-im', '--image', type=str, default='')
    parser.add_argument('-wi', '--width', help='requested cap width', type=int, default=640)
    parser.add_argument('-he', '--height', help='requested cap height', type=int, default=480)
    parser.add_argument(
        '-mdc', '--min_detection_confidence',
        help='min_detection_confidence', type=float, default=0.6,
    )
    parser.add_argument('-dif', '--disable_image_flip', help='disable image flip', action='store_true')
    parser.add_argument('-o', '--output', help='output video path', type=str, default='output.mp4')
    parser.add_argument('-nv', '--no_video', help='do not write an output video', action='store_true')
    parser.add_argument('--headless', help='do not open a window (no keyboard control)', action='store_true')
    parser.add_argument(
        '-tmd', '--track_max_distance',
        help='max palm-centre movement (px) between frames for the same track',
        type=float, default=100.0,
    )
    parser.add_argument(
        '-tmm', '--track_max_missed',
        help='frames a track survives without a matching detection',
        type=int, default=5,
    )

    parser.add_argument('--keypoint_model', type=str, default='',
                        help='ONNX hand-sign classifier (default: shipped model)')
    parser.add_argument('--point_history_model', type=str, default='',
                        help='ONNX finger-gesture classifier (default: shipped LSTM model)')

    return parser.parse_args()


def read_labels(path):
    with open(path, encoding='utf-8-sig') as f:
        return [row[0] for row in csv.reader(f)]


def put_outlined_text(image, text, org, scale=0.8, color=(59, 255, 255)):
    cv.putText(image, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv.LINE_AA)
    cv.putText(image, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv.LINE_AA)


def clamp_text_pos(x, y, frame_w, frame_h, dy_min, dy_max):
    tx = min(max(x, 10), frame_w - 120)
    ty = min(max(y, dy_min), frame_h - dy_max)
    return int(tx), int(ty)


def palms_to_rects(hands, frame_w, frame_h):
    """Convert PalmDetection output to pixel rects [cx, cy, w, h, degree].

    sqn_rr_size is relative to max(frame_w, frame_h) (the padded square the
    detector saw), so the crop is square in pixels for any aspect ratio.
    """
    square = max(frame_w, frame_h)
    rects = []
    for sqn_rr_size, rotation, cxn, cyn in hands:
        cx = cxn * frame_w
        cy = cyn * frame_h
        side = sqn_rr_size * square
        xmin = max(0, int(cx - side / 2))
        xmax = min(frame_w, int(cx + side / 2))
        ymin = max(0, int(cy - side / 2))
        ymax = min(frame_h, int(cy + side / 2))
        rects.append([int(cx), int(cy), (xmax - xmin), (ymax - ymin), degrees(rotation)])
    return np.asarray(rects, dtype=np.float32).reshape(-1, 5)


def main():
    args = get_args()

    cap_device = args.image if args.image else args.device
    lines_hand = [
        [0,1],[1,2],[2,3],[3,4],
        [0,5],[5,6],[6,7],[7,8],
        [5,9],[9,10],[10,11],[11,12],
        [9,13],[13,14],[14,15],[15,16],
        [13,17],[17,18],[18,19],[19,20],[0,17],
    ]

    # Camera ###################################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    cap_fps = cap.get(cv.CAP_PROP_FPS)
    if not cap_fps or cap_fps <= 0 or cap_fps > 240:
        cap_fps = 30.0
    video_writer = None  # created on the first frame, using the real frame size

    # Models ###################################################################
    palm_detection = PalmDetection(score_threshold=args.min_detection_confidence)
    hand_landmark = HandLandmark()
    keypoint_classifier = (
        KeyPointClassifier(model_path=args.keypoint_model) if args.keypoint_model else KeyPointClassifier())
    point_history_classifier = (
        PointHistoryClassifier(model_path=args.point_history_model)
        if args.point_history_model else PointHistoryClassifier())

    keypoint_classifier_labels = read_labels(
        os.path.join(APP_DIR, 'model/keypoint_classifier/keypoint_classifier_label.csv'))
    point_history_classifier_labels = read_labels(
        os.path.join(APP_DIR, 'model/point_history_classifier/point_history_classifier_label.csv'))

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    tracker = HandTracker(
        max_distance=args.track_max_distance,
        max_missed=args.track_max_missed,
    )

    # Per-track state, always keyed by the tracker's integer track id.
    point_history = {}           # trackid -> deque[[x, y]] of index fingertip
    finger_gesture_history = {}  # trackid -> deque[int] of gesture ids

    mode = 0
    auto = False
    prev_number = -1
    image = None

    while True:
        fps = cvFpsCalc.get()

        # Keys (ESC: quit). For a still image, wait after the first frame is shown.
        if args.headless:
            key = -1
        else:
            key = cv.waitKey(0) if (args.image and image is not None) else cv.waitKey(1)
        if key == 27:  # ESC
            break
        number, mode, auto, prev_number = select_mode(key, mode, auto, prev_number)

        ret, image = cap.read()
        if not ret:
            break
        image = image if args.disable_image_flip else cv.flip(image, 1)
        debug_image = image.copy()
        frame_h, frame_w = image.shape[:2]

        if video_writer is None and not args.no_video:
            video_writer = cv.VideoWriter(
                args.output,
                cv.VideoWriter_fourcc(*'mp4v'),
                cap_fps,
                (frame_w, frame_h),
            )

        # PalmDetection ########################################################
        hands = palm_detection(image)
        rects = palms_to_rects(hands, frame_w, frame_h)

        # Draw detector boxes (red: rotated, orange: upright)
        for rect in rects:
            box = cv.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4])).astype(np.intp)
            cv.drawContours(debug_image, [box], 0, (0, 0, 255), 2, cv.LINE_AA)
            rcx, rcy = int(rect[0]), int(rect[1])
            half_w, half_h = int(rect[2] // 2), int(rect[3] // 2)
            x1, y1, x2, y2 = rcx - half_w, rcy - half_h, rcx + half_w, rcy + half_h
            put_outlined_text(debug_image, f'{y2-y1}x{x2-x1}',
                              clamp_text_pos(x1, y1 - 15, frame_w, frame_h, 45, 20))
            cv.rectangle(debug_image, (x1, y1), (x2, y2), (0, 128, 255), 2, cv.LINE_AA)
            cv.circle(debug_image, (rcx, rcy), 3, (0, 255, 255), -1)

        # HandLandmark #########################################################
        hand_landmarks = np.zeros((0, 21, 2), np.int32)
        size_leftrights = np.zeros((0, 3), np.float32)
        keep = np.zeros((0,), np.int64)
        if len(rects) > 0:
            crops = rotate_and_crop_rectangle(
                image=image,
                rects_tmp=rects,
                operation_when_cropping_out_of_range='padding',
            )
            hand_landmarks, size_leftrights, keep = hand_landmark(images=crops, rects=rects)

        # Tracking: only hands that passed the landmark score are tracked, so
        # index i of hand_landmarks / size_leftrights / kept_rects / track_ids
        # always refers to the same physical hand.
        kept_rects = rects[keep] if len(keep) > 0 else np.zeros((0, 5), np.float32)
        track_ids = tracker.update(kept_rects[:, :2])
        alive = set(tracker.active_ids)
        for state in (point_history, finger_gesture_history):
            for tid in [t for t in state if t not in alive]:
                del state[tid]

        # Hand sign classification (batched) ###################################
        pre_processed_landmarks = [pre_process_landmark(lm) for lm in hand_landmarks]
        hand_sign_ids = (
            keypoint_classifier(np.asarray(pre_processed_landmarks, dtype=np.float32))
            if len(pre_processed_landmarks) > 0 else []
        )

        # Fingertip history for the current frame (no one-frame lag)
        for tid, landmark, sign_id in zip(track_ids, hand_landmarks, hand_sign_ids):
            hist = point_history.setdefault(tid, deque(maxlen=HISTORY_LENGTH))
            if int(sign_id) == POINTER_SIGN_ID:
                hist.append([int(landmark[INDEX_FINGER_TIP][0]), int(landmark[INDEX_FINGER_TIP][1])])
            else:
                hist.append([0, 0])

        pre_processed_histories = {
            tid: pre_process_point_history(frame_w, frame_h, point_history[tid])
            for tid in track_ids
        }

        # Finger gesture classification (batched, full histories only) ########
        full_tids = [t for t in track_ids if len(pre_processed_histories[t]) == HISTORY_LENGTH * 2]
        if len(full_tids) > 0:
            gesture_ids = point_history_classifier(
                np.asarray([pre_processed_histories[t] for t in full_tids], dtype=np.float32)
            )
            for tid, gid in zip(full_tids, gesture_ids):
                finger_gesture_history.setdefault(
                    tid, deque(maxlen=GESTURE_HISTORY_LENGTH)).append(int(gid))

        # Training data logging (one row per hand) ############################
        for i, tid in enumerate(track_ids):
            logging_csv(number, mode, pre_processed_landmarks[i], pre_processed_histories[tid])

        # Draw per hand ########################################################
        for i, tid in enumerate(track_ids):
            landmark = hand_landmarks[i]
            rotated_w, _, lr = size_leftrights[i]
            rect = kept_rects[i]
            x1 = int(rect[0]) - int(rect[2] // 2)
            y1 = int(rect[1]) - int(rect[3] // 2)

            radius = int(1 + (rotated_w / 400) * 5)
            lines = np.asarray([[landmark[p] for p in line] for line in lines_hand], dtype=np.int32)
            cv.polylines(debug_image, lines, False, (255, 0, 0), radius, cv.LINE_AA)
            for x, y in landmark:
                cv.circle(debug_image, (int(x), int(y)), radius, (0, 128, 255), -1)

            lr = int(lr) if args.disable_image_flip else 1 - int(lr)
            handedness = 'Left ' if lr == 0 else 'Right'
            sign = keypoint_classifier_labels[int(hand_sign_ids[i])]
            put_outlined_text(debug_image, f'trackid:{tid} {handedness} {sign}',
                              clamp_text_pos(x1, y1 - 70, frame_w, frame_h, 20, 70))

            if tid in finger_gesture_history:
                most_common = Counter(finger_gesture_history[tid]).most_common(1)[0][0]
                put_outlined_text(debug_image, point_history_classifier_labels[most_common],
                                  clamp_text_pos(x1, y1 - 45, frame_w, frame_h, 20, 45))

        visible_history = {t: point_history[t] for t in track_ids if t in point_history}
        debug_image = draw_point_history(debug_image, visible_history)
        debug_image = draw_info(debug_image, fps, mode, number, auto)

        if not args.headless:
            cv.imshow('Hand Gesture Recognition', debug_image)
        if video_writer is not None:
            video_writer.write(debug_image)

    if video_writer is not None:
        video_writer.release()
    cap.release()
    if not args.headless:
        cv.destroyAllWindows()


def select_mode(key, mode, auto=False, prev_number=-1):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
        prev_number = number
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    if key == 97:   # a
        auto = not auto
    if auto == True:
        if prev_number != -1:
            number = prev_number
    else:
        prev_number = -1

    return number, mode, auto, prev_number




def pre_process_landmark(landmark_list):
    """21x(x, y) pixel landmarks -> 42 floats relative to the wrist, scaled to [-1, 1]."""
    if len(landmark_list) == 0:
        return []
    base_x, base_y = landmark_list[0][0], landmark_list[0][1]
    flat = list(itertools.chain.from_iterable(
        [float(p[0] - base_x), float(p[1] - base_y)] for p in landmark_list
    ))
    max_value = max(map(abs, flat))
    if max_value == 0:
        return [0.0] * len(flat)
    return [v / max_value for v in flat]


def pre_process_point_history(image_width, image_height, points):
    """Fingertip history of ONE track -> flat [rx, ry, ...] relative to its first point,
    normalised by frame size."""
    if len(points) == 0:
        return []
    base_x, base_y = points[0][0], points[0][1]
    return list(itertools.chain.from_iterable(
        [(p[0] - base_x) / image_width, (p[1] - base_y) / image_height] for p in points
    ))


def logging_csv(number, mode, landmark_list, point_history):
    """Append one training row in the format the notebooks read:
    keypoint.csv: label, 42 features; point_history.csv: label, 32 features.
    Point histories are only logged once they are full length."""
    if not (0 <= number <= 9):
        return
    if mode == 1 and len(landmark_list) == 42:
        with open(KEYPOINT_CSV, 'a', newline="") as f:
            csv.writer(f).writerow([number, *landmark_list])
    if mode == 2 and len(point_history) == HISTORY_LENGTH * 2:
        with open(POINT_HISTORY_CSV, 'a', newline="") as f:
            csv.writer(f).writerow([number, *point_history])


def draw_info_text(
    image,
    brect,
    handedness,
    hand_sign_text,
    finger_gesture_text
):
    info_text = handedness
    if hand_sign_text != "":
        info_text = f'{handedness}:{hand_sign_text}'
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    if finger_gesture_text != "":
        cv.putText(
            image,
            f'Finger Gesture:{finger_gesture_text}',
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            f'Finger Gesture:{finger_gesture_text}',
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv.LINE_AA,
        )

    return image


def draw_point_history(image, point_history):
    _ = [
        cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2) \
            for trackid, points in point_history.items() \
                for index, point in enumerate(points) if point[0] != 0 and point[1] != 0
    ]
    return image


def draw_info(image, fps, mode, number, auto):
    cv.putText(
        image,
        f'FPS:{str(fps)}',
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        f'FPS:{str(fps)}',
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(
            image,
            f'MODE:{mode_string[mode - 1]}',
            (10, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        if 0 <= number <= 9:
            cv.putText(
                image,
                f'NUM:{str(number)}',
                (10, 110),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
    cv.putText(
        image,
        f'AUTO:{str(auto)}',
        (10, 130),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )
    return image



if __name__ == '__main__':
    main()
