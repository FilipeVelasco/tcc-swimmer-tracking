"""
Camada de rastreamento (tracking).

Associação primária: ByteTrack (via `supervision`), baseada em movimento/IoU
+ filtro de Kalman — é a mesma abordagem leve documentada no TCC1.

Fallback de reassociação por embedding (documentado no TCC1 como
"fallback framing for embedding-based association"): quando o ByteTrack
não consegue casar uma detecção nova com nenhuma track ativa (ex.: nadador
saiu de cena momentaneamente, oclusão entre raias, etc.), tentamos resgatar
o ID original comparando um descritor de aparência leve (histograma de cor
HSV — propositalmente leve, sem rede de embedding pesada) contra as tracks
perdidas recentemente. Só se a similaridade de cosseno superar o limiar é
que o ID antigo é restaurado; caso contrário, o ByteTrack segue seu fluxo
normal e cria uma nova track.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv

from src.utils.config import TrackerConfig


@dataclass
class LostTrack:
    track_id: int
    embedding: np.ndarray
    lost_at_frame: int


@dataclass
class TrackHistory:
    """Histórico simples de posições por track_id, usado depois pela pose/stroke counting."""
    positions: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = field(default_factory=dict)

    def update(self, frame_idx: int, track_id: int, bbox_xyxy: tuple[float, float, float, float]) -> None:
        self.positions.setdefault(track_id, []).append((frame_idx, bbox_xyxy))

    def get(self, track_id: int) -> list[tuple[int, tuple[float, float, float, float]]]:
        return self.positions.get(track_id, [])


class AppearanceReID:
    """Descritor de aparência leve (histograma HSV) + resgate de tracks perdidas."""

    def __init__(self, bins: list[int], max_lost_age_frames: int, similarity_threshold: float):
        self.bins = bins
        self.max_lost_age_frames = max_lost_age_frames
        self.similarity_threshold = similarity_threshold
        self._lost: dict[int, LostTrack] = {}
        self._next_synthetic_check = 0

    @staticmethod
    def compute_embedding(frame: np.ndarray, bbox_xyxy: np.ndarray, bins: list[int]) -> np.ndarray:
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox_xyxy]
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(int(np.prod(bins)), dtype=np.float32)
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype(np.float32)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def mark_lost(self, track_id: int, embedding: np.ndarray, frame_idx: int) -> None:
        self._lost[track_id] = LostTrack(track_id, embedding, frame_idx)

    def try_rescue(self, embedding: np.ndarray, frame_idx: int) -> int | None:
        """Tenta casar um embedding novo com uma track perdida recentemente. Retorna o ID resgatado ou None."""
        self._prune(frame_idx)
        best_id, best_sim = None, self.similarity_threshold
        for lost in self._lost.values():
            sim = self.cosine_similarity(embedding, lost.embedding)
            if sim > best_sim:
                best_sim, best_id = sim, lost.track_id
        if best_id is not None:
            del self._lost[best_id]
        return best_id

    def _prune(self, frame_idx: int) -> None:
        expired = [tid for tid, lt in self._lost.items()
                   if frame_idx - lt.lost_at_frame > self.max_lost_age_frames]
        for tid in expired:
            del self._lost[tid]


class SwimmerTracker:
    """Combina ByteTrack (motion/IoU) com o fallback de re-ID por aparência."""

    def __init__(self, config: TrackerConfig):
        self.config = config
        # NOTA: `sv.ByteTrack` está marcado como deprecated (remoção prevista p/ v0.31.0
        # do `supervision`, sem substituto direto anunciado até o momento — possivelmente
        # migrará para o pacote `trackers` da Roboflow). Fixamos a versão em
        # requirements.txt e silenciamos o aviso aqui; ao atualizar o supervision no
        # futuro, revisar esta classe primeiro.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.byte_track = sv.ByteTrack(
                track_activation_threshold=config.track_activation_threshold,
                lost_track_buffer=config.lost_track_buffer,
                minimum_matching_threshold=config.minimum_matching_threshold,
                frame_rate=config.frame_rate,
            )
        self.reid = AppearanceReID(
            bins=config.reid_fallback.histogram_bins,
            max_lost_age_frames=config.reid_fallback.max_lost_age_frames,
            similarity_threshold=config.reid_fallback.similarity_threshold,
        ) if config.reid_fallback.enabled else None
        self.history = TrackHistory()
        self._active_ids_prev_frame: set[int] = set()
        self._embeddings_by_id: dict[int, np.ndarray] = {}

    def update(self, frame: np.ndarray, detections: sv.Detections, frame_idx: int) -> sv.Detections:
        tracked = self.byte_track.update_with_detections(detections)

        if self.reid is not None:
            tracked = self._apply_reid_fallback(frame, tracked, frame_idx)

        current_ids = set(tracked.tracker_id.tolist()) if tracked.tracker_id is not None else set()

        # tracks que sumiram desde o frame anterior -> guardar embedding para possível resgate futuro
        if self.reid is not None:
            newly_lost = self._active_ids_prev_frame - current_ids
            for tid in newly_lost:
                emb = self._embeddings_by_id.get(tid)
                if emb is not None:
                    self.reid.mark_lost(tid, emb, frame_idx)

        # atualizar histórico e embeddings das tracks ativas
        if tracked.tracker_id is not None:
            for bbox, tid in zip(tracked.xyxy, tracked.tracker_id):
                self.history.update(frame_idx, int(tid), tuple(bbox.tolist()))
                if self.reid is not None:
                    self._embeddings_by_id[int(tid)] = self.reid.compute_embedding(frame, bbox, self.reid.bins)

        self._active_ids_prev_frame = current_ids
        return tracked

    def _apply_reid_fallback(self, frame: np.ndarray, tracked: sv.Detections, frame_idx: int) -> sv.Detections:
        """
        ByteTrack já atribui IDs novos para detecções não casadas. Aqui, para cada ID
        que é 'novo' (nunca visto antes nesta sessão de tracking), verificamos se ele
        corresponde, por aparência, a uma track recentemente perdida — se sim,
        substituímos pelo ID antigo (fallback de embedding).
        """
        if tracked.tracker_id is None or len(tracked) == 0:
            return tracked

        new_ids = tracked.tracker_id.copy()
        for i, tid in enumerate(tracked.tracker_id):
            tid = int(tid)
            if tid in self._embeddings_by_id or tid in self._active_ids_prev_frame:
                continue  # já é uma track conhecida, não precisa de resgate
            embedding = self.reid.compute_embedding(frame, tracked.xyxy[i], self.reid.bins)
            rescued_id = self.reid.try_rescue(embedding, frame_idx)
            if rescued_id is not None:
                new_ids[i] = rescued_id
        tracked.tracker_id = new_ids
        return tracked
