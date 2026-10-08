"""
Gera um vídeo sintético "hipotético" simulando nadadores vistos de cima (overhead),
junto com o ground-truth de tracking no formato MOTChallenge.

Objetivo: permitir testar o pipeline inteiro (tracking, pose/stroke, avaliação)
end-to-end SEM depender do vídeo real nem dos pesos treinados de verdade.
Cada "nadador" é um retângulo colorido que se move em raia própria e possui
2 keypoints (punho esq./dir.) oscilando senoidalmente para simular braçadas.

Formato do GT (data/annotations/gt_mot.txt), padrão MOTChallenge:
    frame, id, x, y, w, h, conf, class, visibility
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

random.seed(42)
np.random.seed(42)


@dataclass
class SyntheticSwimmer:
    track_id: int
    lane_y: float
    color: tuple[int, int, int]
    speed_px_s: float
    stroke_freq_hz: float
    box_w: int
    box_h: int
    start_frame: int
    end_frame: int
    x0: float

    def bbox_at(self, frame_idx: int, fps: int) -> tuple[float, float, float, float] | None:
        if not (self.start_frame <= frame_idx <= self.end_frame):
            return None
        t = (frame_idx - self.start_frame) / fps
        cx = self.x0 + self.speed_px_s * t
        cy = self.lane_y
        return (cx - self.box_w / 2, cy - self.box_h / 2, self.box_w, self.box_h)

    def wrist_offsets(self, frame_idx: int, fps: int) -> tuple[float, float]:
        """Retorna deslocamento vertical (em px) dos 2 punhos, simulando a braçada."""
        t = (frame_idx - self.start_frame) / fps
        phase = 2 * math.pi * self.stroke_freq_hz * t
        amp = self.box_h * 0.35
        left = amp * math.sin(phase)
        right = amp * math.sin(phase + math.pi)  # braços alternados
        return left, right


def build_swimmers(n_lanes: int, width: int, height: int, fps: int, duration_s: int) -> list[SyntheticSwimmer]:
    total_frames = fps * duration_s
    lane_height = height / (n_lanes + 1)
    swimmers = []
    palette = [
        (66, 135, 245), (245, 66, 66), (66, 245, 129), (245, 194, 66),
        (188, 66, 245), (66, 245, 233), (245, 66, 173), (150, 245, 66),
    ]
    for i in range(n_lanes):
        lane_y = lane_height * (i + 1)
        speed = random.uniform(90, 140)  # px/s
        stroke_freq = random.uniform(0.5, 0.9)  # Hz (braçadas/segundo, por braço)
        box_w, box_h = random.randint(70, 90), random.randint(28, 38)
        start_frame = random.randint(0, fps * 2)
        # duração de travessia da raia dada a velocidade
        travel_frames = int((width - box_w) / speed * fps)
        end_frame = min(total_frames - 1, start_frame + travel_frames)
        swimmers.append(
            SyntheticSwimmer(
                track_id=i + 1,
                lane_y=lane_y,
                color=palette[i % len(palette)],
                speed_px_s=speed,
                stroke_freq_hz=stroke_freq,
                box_w=box_w,
                box_h=box_h,
                start_frame=start_frame,
                end_frame=end_frame,
                x0=-box_w / 2,
            )
        )
    return swimmers


def render_frame(width: int, height: int, swimmers: list[SyntheticSwimmer], frame_idx: int, fps: int) -> np.ndarray:
    frame = np.full((height, width, 3), (40, 90, 30), dtype=np.uint8)  # "água" verde-azulada escura
    # raias
    n_lanes = len(swimmers)
    lane_height = height / (n_lanes + 1)
    for i in range(1, n_lanes + 1):
        y = int(lane_height * i - lane_height / 2)
        cv2.line(frame, (0, y), (width, y), (200, 200, 200), 1, lineType=cv2.LINE_AA)

    for sw in swimmers:
        bbox = sw.bbox_at(frame_idx, fps)
        if bbox is None:
            continue
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        cv2.ellipse(frame, (int(cx), int(cy)), (int(w / 2), int(h / 2)), 0, 0, 360, sw.color, -1)
        left_off, right_off = sw.wrist_offsets(frame_idx, fps)
        cv2.circle(frame, (int(cx - w * 0.35), int(cy + left_off)), 3, (255, 255, 255), -1)
        cv2.circle(frame, (int(cx + w * 0.35), int(cy + right_off)), 3, (255, 255, 255), -1)
    return frame


def generate(
    output_video: str = "data/synthetic/synthetic_video.mp4",
    output_gt: str = "data/annotations/gt_mot.txt",
    output_keypoints_gt: str = "data/annotations/gt_keypoints.csv",
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration_s: int = 12,
    n_lanes: int = 6,
) -> None:
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    Path(output_gt).parent.mkdir(parents=True, exist_ok=True)

    swimmers = build_swimmers(n_lanes, width, height, fps, duration_s)
    total_frames = fps * duration_s

    writer = cv2.VideoWriter(
        output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    gt_rows = []
    kp_rows = []
    for frame_idx in range(total_frames):
        frame = render_frame(width, height, swimmers, frame_idx, fps)
        writer.write(frame)
        for sw in swimmers:
            bbox = sw.bbox_at(frame_idx, fps)
            if bbox is None:
                continue
            x, y, w, h = bbox
            # MOTChallenge é 1-indexado em frame
            gt_rows.append([frame_idx + 1, sw.track_id, round(x, 2), round(y, 2),
                             round(w, 2), round(h, 2), 1, 1, 1.0])
            left_off, right_off = sw.wrist_offsets(frame_idx, fps)
            cx, cy = x + w / 2, y + h / 2
            kp_rows.append([
                frame_idx + 1, sw.track_id,
                round(cx - w * 0.35, 2), round(cy + left_off, 2),
                round(cx + w * 0.35, 2), round(cy + right_off, 2),
            ])
    writer.release()

    with open(output_gt, "w", newline="") as f:
        csv.writer(f).writerows(gt_rows)

    with open(output_keypoints_gt, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "left_wrist_x", "left_wrist_y", "right_wrist_x", "right_wrist_y"])
        w.writerows(kp_rows)

    print(f"[OK] Vídeo sintético: {output_video} ({total_frames} frames, {n_lanes} nadadores)")
    print(f"[OK] GT de tracking (MOT format): {output_gt} ({len(gt_rows)} linhas)")
    print(f"[OK] GT de keypoints (punhos): {output_keypoints_gt} ({len(kp_rows)} linhas)")


if __name__ == "__main__":
    generate()
