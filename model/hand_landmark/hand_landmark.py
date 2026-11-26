import os
import copy
from typing import (
    Tuple,
    Optional,
    List,
)
import cv2
import onnxruntime
import numpy as np

from utils.utils import keep_aspect_resize_and_pad


class HandLandmark(object):
    def __init__(
        self,
        model_path: Optional[str] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmark_sparse_Nx3x224x224.onnx'),
        class_score_th: Optional[float] = 0.50,
        providers: Optional[List] = [
            # (
            #     'TensorrtExecutionProvider', {
            #         'trt_engine_cache_enable': True,
            #         'trt_engine_cache_path': '.',
            #         'trt_fp16_enable': True,
            #     }
            # ),
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        """HandLandmark

        Parameters
        ----------
        model_path: Optional[str]
            ONNX file path for Hand Landmark

        class_score_th: Optional[float]
            Score threshold. Default: 0.50

        providers: Optional[List]
            Name of onnx execution providers
            Default:
            [
                (
                    'TensorrtExecutionProvider', {
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': '.',
                        'trt_fp16_enable': True,
                    }
                ),
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        """
        # Threshold
        self.class_score_th = class_score_th

        # Model loading
        # Drop providers this onnxruntime build does not have (avoids warnings on CPU-only installs)
        available = onnxruntime.get_available_providers()
        providers = [p for p in providers if (p[0] if isinstance(p, tuple) else p) in available]
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )
        self.providers = self.onnx_session.get_providers()

        self.input_shapes = [
            input.shape for input in self.onnx_session.get_inputs()
        ]
        self.input_names = [
            input.name for input in self.onnx_session.get_inputs()
        ]
        self.output_names = [
            output.name for output in self.onnx_session.get_outputs()
        ]


    def __call__(
        self,
        images: List[np.ndarray],
        rects: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """HandLandmark

        Parameters
        ----------
        images: List[np.ndarray]
            Multiple palm images.

        rects: np.ndarray
            Coordinates, size and angle of the cut palm.
            [boxcount, cx, cy, width, height, angle]

        Returns
        -------
        hand_landmarks: np.ndarray
            Hand landmarks (X,Y) x 21

        rotated_image_size_leftrights: np.ndarray
            Maximum width and height of the perimeter of the rectangle around
            which the bounding box of the detected hand is rotated,
            and flags for left and right hand
            [rotated_image_width, rotated_image_height, left_hand_0_or_right_hand_1]

        keep_indices: np.ndarray
            int64[M] indices into `images`/`rects` of the hands that passed
            class_score_th, in the same order as hand_landmarks. Callers must use
            this to pair results with their per-detection data.
        """
        temp_images = images  # not mutated

        # PreProcess
        inference_images, resized_images, resize_scales_224x224, half_pad_sizes_224x224 = self.__preprocess(
            images=temp_images,
        )

        # Inference
        xyz_x21s, hand_scores, left_hand_0_or_right_hand_1s = self.onnx_session.run(
            self.output_names,
            {input_name: inference_images for input_name in self.input_names},
        )

        # PostProcess
        hand_landmarks, rotated_image_size_leftrights, keep_indices = self.__postprocess(
            resized_images=resized_images,
            resize_scales_224x224=resize_scales_224x224,
            half_pad_sizes_224x224=half_pad_sizes_224x224,
            rects=rects,
            xyz_x21s=xyz_x21s,
            hand_scores=hand_scores,
            left_hand_0_or_right_hand_1s=left_hand_0_or_right_hand_1s,
        )

        return hand_landmarks, rotated_image_size_leftrights, keep_indices


    def __preprocess(
        self,
        images: List[np.ndarray],
        swap: Optional[Tuple[int,int,int]] = (2, 0, 1),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """__preprocess

        Parameters
        ----------
        images: List[np.ndarray]
            Multiple palm images.

        swap: tuple
            HWC to CHW: (2,0,1)
            CHW to HWC: (1,2,0)
            HWC to HWC: (0,1,2)
            CHW to CHW: (0,1,2)

        Returns
        -------
        padded_images: np.ndarray
            Resized and Padding and normalized image. [N,C,H,W]
        """
        temp_images = images  # not mutated

        # Resize + Padding + Normalization + BGR->RGB
        input_h = self.input_shapes[0][2]
        input_w = self.input_shapes[0][3]

        padded_images = []
        resized_images = []
        resize_scales_224x224 = []
        half_pad_sizes_224x224 = []
        for image in temp_images:
            padded_image, resized_image = keep_aspect_resize_and_pad(
                image=image,
                resize_width=input_w,
                resize_height=input_h,
            )
            # reduction_ratio_h = resized_h / original_h
            resize_224x224_scale_h = resized_image.shape[0] / image.shape[0]
            # reduction_ratio_w = resized_w / original_w
            resize_224x224_scale_w = resized_image.shape[1] / image.shape[1]
            resize_scales_224x224.append(
                [
                    resize_224x224_scale_w,
                    resize_224x224_scale_h,
                ]
            )

            pad_h = padded_image.shape[0] - resized_image.shape[0]
            pad_w = padded_image.shape[1] - resized_image.shape[1]
            half_pad_h_224x224 = pad_h // 2
            half_pad_h_224x224 = half_pad_h_224x224 if half_pad_h_224x224 >= 0 else 0
            half_pad_w_224x224 = pad_w // 2
            half_pad_w_224x224 = half_pad_w_224x224 if half_pad_w_224x224 >= 0 else 0
            half_pad_sizes_224x224.append([half_pad_w_224x224, half_pad_h_224x224])

            padded_image = np.divide(padded_image, 255.0)
            padded_image = padded_image[..., ::-1]
            padded_image = padded_image.transpose(swap)
            padded_image = np.ascontiguousarray(
                padded_image,
                dtype=np.float32,
            )

            padded_images.append(padded_image)
            resized_images.append(resized_image)

        return \
            np.asarray(padded_images, dtype=np.float32), \
            resized_images, \
            np.asarray(resize_scales_224x224, dtype=np.float32), \
            np.asarray(half_pad_sizes_224x224, dtype=np.int32)


    def __postprocess(
        self,
        resized_images: List[np.ndarray],
        resize_scales_224x224: np.ndarray,
        half_pad_sizes_224x224: np.ndarray,
        rects: np.ndarray,
        xyz_x21s: np.ndarray,
        hand_scores: np.ndarray,
        left_hand_0_or_right_hand_1s: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """__postprocess

        Parameters
        ----------
        rects: np.ndarray
            [N, cx, cy, (xmax-xmin), (ymax-ymin), degree]

        xyz_x21s: np.ndarray
            float32[N, 63]
            XYZ coordinates. 21 points.

        hand_scores: np.ndarray
            float32[N, 1]
            Hand score.

        left_hand_0_or_right_hand_1s: np.ndarray
            float32[N, 1]
            0: Left hand
            1: Right hand

        Returns
        -------
        extracted_hands: np.ndarray
            Hand landmarks (X,Y) x 21

        rotated_image_size_leftrights: np.ndarray
            Maximum width and height of the perimeter of the rectangle around
            which the bounding box of the detected hand is rotated,
            and flags for left and right hand
            [rotated_image_width, rotated_image_height, left_hand_0_or_right_hand_1]
        """
        hand_landmarks = np.asarray([], dtype=np.int32)
        extracted_hands = []
        rotated_image_size_leftrights = []

        keep = hand_scores[:, 0] > self.class_score_th # hand_score > self.class_score_th
        keep_indices = np.nonzero(keep)[0]
        xyz_x21s = xyz_x21s[keep, :]
        hand_scores = hand_scores[keep, :]
        left_hand_0_or_right_hand_1s = left_hand_0_or_right_hand_1s[keep, :]
        resize_scales_224x224 = resize_scales_224x224[keep, :]
        half_pad_sizes_224x224 = half_pad_sizes_224x224[keep, :]
        rects = rects[keep, :]
        resized_images = [i for (i, k) in zip(resized_images, keep) if k == True]

        for resized_image, resize_scale_224x224, half_pad_size_224x224, rect, xyz_x21, left_hand_0_or_right_hand_1 in \
            zip(resized_images, resize_scales_224x224, half_pad_sizes_224x224, rects, xyz_x21s, left_hand_0_or_right_hand_1s):
            """
            hands: sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y
                cx = int(sqn_rr_center_x * frame_width)
                cy = int(sqn_rr_center_y * frame_height)
                xmin = int((sqn_rr_center_x - (sqn_rr_size / 2))*w)
                xmax = int((sqn_rr_center_x + (sqn_rr_size / 2))*w)
                ymin = int((sqn_rr_center_y - (sqn_rr_size * wh_ratio / 2))*h)
                ymax = int((sqn_rr_center_y + (sqn_rr_size * wh_ratio / 2))*h)
                degree = degrees(rotation)
            rect : cx, cy, (xmax-xmin), (ymax-ymin), degree
                rotation = radians(degree)
                sqn_rr_center_x = cx / frame_width
                sqn_rr_center_y = cy / frame_height
            """
            rrn_lms = xyz_x21
            input_h = self.input_shapes[0][2]
            input_w = self.input_shapes[0][3]
            rrn_lms = rrn_lms / input_h

            rcx = rect[0]
            rcy = rect[1]
            angle = rect[4]

            # Size the crop would have after undoing the 224x224 resize
            # (same rounding as cv2.resize with fx/fy), computed without resizing.
            width = int(round(resized_image.shape[1] / resize_scale_224x224[0]))
            height = int(round(resized_image.shape[0] / resize_scale_224x224[1]))
            rescaled_xy = np.asarray([[v[0], v[1]] for v in zip(rrn_lms[0::3], rrn_lms[1::3])], dtype=np.float32)
            rescaled_xy[:, 0] = (rescaled_xy[:, 0] * input_w - half_pad_size_224x224[0]) / resize_scale_224x224[0]
            rescaled_xy[:, 1] = (rescaled_xy[:, 1] * input_h - half_pad_size_224x224[1]) / resize_scale_224x224[1]
            rescaled_xy = rescaled_xy.astype(np.int32)

            image_center = (width//2, height//2)
            rotation_matrix = cv2.getRotationMatrix2D(image_center, -float(angle), 1)
            abs_cos = abs(rotation_matrix[0,0])
            abs_sin = abs(rotation_matrix[0,1])
            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)
            rotation_matrix[0, 2] += bound_w/2 - image_center[0]
            rotation_matrix[1, 2] += bound_h/2 - image_center[1]

            # Apply the 2x3 affine to all 21 points at once (also NumPy 2.x safe:
            # int() of a 1-element array is an error there).
            homog = np.concatenate([rescaled_xy, np.ones((len(rescaled_xy), 1), np.int32)], axis=1)
            keypoints = (homog @ rotation_matrix.T).astype(np.int32)

            rotated_image_width = bound_w
            rotated_image_height = bound_h
            roatated_hand_half_width = rotated_image_width//2
            roatated_hand_half_height = rotated_image_height//2

            hand_landmarks = np.asarray(keypoints, dtype=np.int32).reshape(-1,2)
            hand_landmarks[..., 0] = hand_landmarks[..., 0] + rcx - roatated_hand_half_width
            hand_landmarks[..., 1] = hand_landmarks[..., 1] + rcy - roatated_hand_half_height
            extracted_hands.append(hand_landmarks)
            rotated_image_size_leftrights.append([rotated_image_width, rotated_image_height, float(np.ravel(left_hand_0_or_right_hand_1)[0])])

        return \
            np.asarray(extracted_hands, dtype=np.int32).reshape(-1, 21, 2), \
            np.asarray(rotated_image_size_leftrights, dtype=np.float32).reshape(-1, 3), \
            keep_indices
