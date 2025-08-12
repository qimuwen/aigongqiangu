import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise SystemExit(
        "mediapipe is required. Install with: pip install mediapipe opencv-python numpy"
    ) from exc


@dataclass
class Features:
    mouth_opening_ratio: float
    eye_opening_ratio: float


@dataclass
class FrameResult:
    frame_index: int
    timestamp_sec: float
    score: float
    features: Features


class ExaggerationScorer:
    def __init__(
        self,
        weight_mouth: float = 0.6,
        weight_eye: float = 0.4,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        self.weight_mouth = weight_mouth
        self.weight_eye = weight_eye
        self.clip_min = clip_min
        self.clip_max = clip_max

    @staticmethod
    def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def _landmark_pairs_xy(landmarks: List[Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        # landmarks are in normalized coordinates [0,1]; distances stay scale-invariant
        idx = {
            # Eyes
            "left_eye_top": 159,
            "left_eye_bottom": 145,
            "left_eye_outer": 33,
            "left_eye_inner": 133,
            "right_eye_top": 386,
            "right_eye_bottom": 374,
            "right_eye_outer": 263,
            "right_eye_inner": 362,
            # Mouth
            "mouth_top_inner": 13,
            "mouth_bottom_inner": 14,
            "mouth_left": 78,
            "mouth_right": 308,
        }
        return {k: landmarks[v] for k, v in idx.items()}

    def compute_features(self, landmarks: List[Tuple[float, float]]) -> Optional[Features]:
        keypoints = self._landmark_pairs_xy(landmarks)

        # Mouth opening ratio (vertical / horizontal)
        mouth_vertical = self._euclidean(
            keypoints["mouth_top_inner"], keypoints["mouth_bottom_inner"]
        )
        mouth_horizontal = self._euclidean(
            keypoints["mouth_left"], keypoints["mouth_right"]
        )
        if mouth_horizontal <= 1e-6:
            return None
        mouth_opening_ratio = mouth_vertical / mouth_horizontal

        # Eye opening ratio: average of left/right (vertical / horizontal)
        left_eye_vertical = self._euclidean(
            keypoints["left_eye_top"], keypoints["left_eye_bottom"]
        )
        left_eye_horizontal = self._euclidean(
            keypoints["left_eye_outer"], keypoints["left_eye_inner"]
        )
        right_eye_vertical = self._euclidean(
            keypoints["right_eye_top"], keypoints["right_eye_bottom"]
        )
        right_eye_horizontal = self._euclidean(
            keypoints["right_eye_outer"], keypoints["right_eye_inner"]
        )
        if left_eye_horizontal <= 1e-6 or right_eye_horizontal <= 1e-6:
            return None
        left_ratio = left_eye_vertical / left_eye_horizontal
        right_ratio = right_eye_vertical / right_eye_horizontal
        eye_opening_ratio = (left_ratio + right_ratio) / 2.0

        return Features(
            mouth_opening_ratio=mouth_opening_ratio,
            eye_opening_ratio=eye_opening_ratio,
        )

    def compute_score(self, current: Features, baseline: Features) -> float:
        # Positive deviation from baseline; clamp negative to zero
        mouth_delta = max(0.0, current.mouth_opening_ratio - baseline.mouth_opening_ratio)
        eye_delta = max(0.0, current.eye_opening_ratio - baseline.eye_opening_ratio)

        # Rough normalization factors so typical expressive frames fall into ~[0,1]
        # You may tune these for your data
        normalized_mouth = mouth_delta / 0.30  # 0.30 mouth ratio increase is "very open"
        normalized_eye = eye_delta / 0.20      # 0.20 eye ratio increase is "very wide"

        raw = self.weight_mouth * normalized_mouth + self.weight_eye * normalized_eye
        return float(np.clip(raw, self.clip_min, self.clip_max))


def _extract_landmarks(
    face_mesh: "mp.solutions.face_mesh.FaceMesh",
    frame_bgr: np.ndarray,
) -> Optional[List[Tuple[float, float]]]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    # Use the most prominent face (index 0)
    lm = result.multi_face_landmarks[0]
    return [(p.x, p.y) for p in lm.landmark]


def analyze_image(image_path: str) -> FrameResult:
    scorer = ExaggerationScorer()
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        landmarks = _extract_landmarks(face_mesh, image)
        if landmarks is None:
            raise RuntimeError("No face detected in the image.")
        features = ExaggerationScorer().compute_features(landmarks)
        if features is None:
            raise RuntimeError("Failed to compute features from landmarks.")

        # For single image, assume a neutral baseline smaller than current to show relative intensity
        baseline = Features(
            mouth_opening_ratio=max(0.0, features.mouth_opening_ratio - 0.1),
            eye_opening_ratio=max(0.0, features.eye_opening_ratio - 0.05),
        )
        score = scorer.compute_score(features, baseline)
        return FrameResult(frame_index=0, timestamp_sec=0.0, score=score, features=features)


def _merge_segments(
    frames: List[FrameResult], threshold: float, fps: float, min_duration_sec: float
) -> List[Tuple[float, float]]:
    segments: List[Tuple[float, float]] = []
    start_idx: Optional[int] = None

    for i, fr in enumerate(frames):
        if fr.score >= threshold and start_idx is None:
            start_idx = i
        if (fr.score < threshold or i == len(frames) - 1) and start_idx is not None:
            end_idx = i if fr.score < threshold else i
            start_t = frames[start_idx].timestamp_sec
            end_t = frames[end_idx].timestamp_sec
            if end_t - start_t >= min_duration_sec:
                segments.append((start_t, end_t))
            start_idx = None
    return segments


def analyze_video(
    video_path: str,
    stride: int = 1,
    baseline_frames: int = 30,
    threshold: float = 0.4,
    min_duration_sec: float = 0.5,
    csv_out: Optional[str] = None,
    out_video_path: Optional[str] = None,
) -> Tuple[List[FrameResult], List[Tuple[float, float]]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    scorer = ExaggerationScorer()

    mp_face_mesh = mp.solutions.face_mesh
    results: List[FrameResult] = []

    writer: Optional[cv2.VideoWriter] = None
    if out_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_video_path, fourcc, fps / max(1, stride), (width, height))

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        frame_index = 0
        baseline_values: List[Features] = []
        baseline: Optional[Features] = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % stride != 0:
                frame_index += 1
                continue

            timestamp_sec = frame_index / fps
            landmarks = _extract_landmarks(face_mesh, frame)
            if landmarks is None:
                frame_index += 1
                continue

            features = scorer.compute_features(landmarks)
            if features is None:
                frame_index += 1
                continue

            if baseline is None:
                baseline_values.append(features)
                if len(baseline_values) >= baseline_frames:
                    baseline = Features(
                        mouth_opening_ratio=float(np.median([f.mouth_opening_ratio for f in baseline_values])),
                        eye_opening_ratio=float(np.median([f.eye_opening_ratio for f in baseline_values])),
                    )
                score = 0.0
            else:
                score = scorer.compute_score(features, baseline)

            result = FrameResult(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                score=score,
                features=features,
            )
            results.append(result)

            if writer is not None:
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (330, 120), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                cv2.putText(frame, f"Score: {score:.2f}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {features.mouth_opening_ratio:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                writer.write(frame)

            frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    segments = _merge_segments(results, threshold=threshold, fps=fps, min_duration_sec=min_duration_sec)

    if csv_out is not None:
        os.makedirs(os.path.dirname(csv_out), exist_ok=True) if os.path.dirname(csv_out) else None
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_index", "timestamp_sec", "score", "mouth_opening_ratio", "eye_opening_ratio"])
            for r in results:
                w.writerow([
                    r.frame_index,
                    f"{r.timestamp_sec:.3f}",
                    f"{r.score:.4f}",
                    f"{r.features.mouth_opening_ratio:.6f}",
                    f"{r.features.eye_opening_ratio:.6f}",
                ])

    return results, segments


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute expression exaggeration score from image or video using MediaPipe FaceMesh.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="Path to an image file")
    src.add_argument("--video", type=str, help="Path to a video file")
    p.add_argument("--stride", type=int, default=1, help="Process every Nth frame for video (default: 1)")
    p.add_argument("--baseline-frames", type=int, default=30, help="Number of initial frames to estimate neutral baseline (video)")
    p.add_argument("--threshold", type=float, default=0.4, help="Score threshold (0-1) to flag expressive segments (video)")
    p.add_argument("--min-duration", type=float, default=0.5, help="Minimum duration in seconds for a segment (video)")
    p.add_argument("--csv-out", type=str, default=None, help="Optional CSV output path for per-frame results (video)")
    p.add_argument("--out-video", type=str, default=None, help="Optional annotated output video path (video)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.image:
        res = analyze_image(args.image)
        print(f"Exaggeration score (image): {res.score:.3f}")
        print(f"Mouth opening ratio: {res.features.mouth_opening_ratio:.4f}")
        print(f"Eye opening ratio:   {res.features.eye_opening_ratio:.4f}")
        return

    results, segments = analyze_video(
        video_path=args.video,
        stride=args.stride,
        baseline_frames=args.baseline_frames,
        threshold=args.threshold,
        min_duration_sec=args.min_duration,
        csv_out=args.csv_out,
        out_video_path=args.out_video,
    )

    if results:
        avg_score = float(np.mean([r.score for r in results]))
        print(f"Average score: {avg_score:.3f}")

    if segments:
        print("Expressive segments (start_sec, end_sec):")
        for s, e in segments:
            print(f"- {s:.2f} -> {e:.2f}")
    else:
        print("No segments above threshold.")


if __name__ == "__main__":
    main()