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


@dataclass
class LaneTrackState:
    lane_id: int
    last_frame: int
    bbox: tuple[float, float, float, float]


@dataclass
class PersistentTrackState:
    logical_id: int
    raw_id: int
    lane_id: int
    bbox: np.ndarray
    embedding: np.ndarray
    velocity: np.ndarray
    last_frame: int


class PersistentIdentityManager:
    """Keeps logical identities across temporary ByteTrack ID fragmentation."""

    def __init__(self, config: TrackerConfig):
        self.config = config
        self.states: dict[int, PersistentTrackState] = {}
        self.raw_to_logical: dict[int, int] = {}
        self.next_id = 1
        self.reassociations = 0
        self.new_identities = 0

    def assign(
        self,
        raw_id: int,
        lane_id: int,
        bbox: np.ndarray,
        embedding: np.ndarray,
        frame_idx: int,
        reserved: set[int],
    ) -> int:
        known_id = self.raw_to_logical.get(raw_id)
        if known_id is not None and known_id not in reserved:
            state = self.states.get(known_id)
            if state is not None and self._lane_compatible(state.lane_id, lane_id):
                self._update(state, raw_id, lane_id, bbox, embedding, frame_idx)
                return known_id

        candidate = self._best_candidate(lane_id, bbox, embedding, frame_idx, reserved)
        if candidate is None:
            logical_id = self.next_id
            self.next_id += 1
            self.new_identities += 1
            self.states[logical_id] = PersistentTrackState(
                logical_id=logical_id,
                raw_id=raw_id,
                lane_id=lane_id,
                bbox=bbox.copy(),
                embedding=embedding.copy(),
                velocity=np.zeros(2, dtype=np.float32),
                last_frame=frame_idx,
            )
        else:
            logical_id = candidate
            self.reassociations += 1
            self._update(self.states[logical_id], raw_id, lane_id, bbox, embedding, frame_idx)
        self.raw_to_logical[raw_id] = logical_id
        return logical_id

    def _best_candidate(
        self,
        lane_id: int,
        bbox: np.ndarray,
        embedding: np.ndarray,
        frame_idx: int,
        reserved: set[int],
    ) -> int | None:
        best: tuple[float, int] | None = None
        for logical_id, state in self.states.items():
            if logical_id in reserved or frame_idx - state.last_frame > self.config.identity_max_gap_frames:
                continue
            if not self._lane_compatible(state.lane_id, lane_id):
                continue
            gap = max(1, frame_idx - state.last_frame)
            predicted = self._predicted_center(state, gap)
            current_center = self._center(bbox)
            diagonal = max(float(np.linalg.norm([bbox[2] - bbox[0], bbox[3] - bbox[1]])), 1.0)
            distance = float(np.linalg.norm(current_center - predicted))
            distance_limit = self.config.identity_distance_factor * diagonal * (1 + 0.15 * gap)
            if distance > distance_limit:
                continue
            iou = self._iou(state.bbox, bbox)
            appearance = AppearanceReID.cosine_similarity(embedding, state.embedding)
            if iou < self.config.identity_min_iou and appearance < self.config.identity_min_appearance_similarity:
                continue
            score = distance / distance_limit + 0.35 * (1 - iou) + 0.35 * (1 - appearance)
            if best is None or score < best[0]:
                best = (score, logical_id)
        return best[1] if best else None

    def _update(
        self, state: PersistentTrackState, raw_id: int, lane_id: int,
        bbox: np.ndarray, embedding: np.ndarray, frame_idx: int,
    ) -> None:
        gap = max(1, frame_idx - state.last_frame)
        velocity = (self._center(bbox) - self._center(state.bbox)) / gap
        state.velocity = 0.7 * state.velocity + 0.3 * velocity
        state.raw_id = raw_id
        state.lane_id = lane_id if lane_id >= 0 else state.lane_id
        state.bbox = bbox.copy()
        state.embedding = 0.7 * state.embedding + 0.3 * embedding
        state.last_frame = frame_idx

    @staticmethod
    def _center(bbox: np.ndarray) -> np.ndarray:
        return np.asarray([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

    @staticmethod
    def _iou(first: np.ndarray, second: np.ndarray) -> float:
        x1, y1 = np.maximum(first[:2], second[:2])
        x2, y2 = np.minimum(first[2:], second[2:])
        intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
        first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
        second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _lane_compatible(first: int, second: int) -> bool:
        return first < 0 or second < 0 or first == second

    def _predicted_center(self, state: PersistentTrackState, gap: int) -> np.ndarray:
        return self._center(state.bbox) + state.velocity * gap


class LaneAwareIdentity:
    """Maintains logical IDs and rejects impossible cross-lane switches."""

    def __init__(self, switch_tolerance_frames: int, max_gap_frames: int, distance_factor: float):
        self.switch_tolerance_frames = switch_tolerance_frames
        self.max_gap_frames = max_gap_frames
        self.distance_factor = distance_factor
        self._raw_to_logical: dict[int, int] = {}
        self._logical_state: dict[int, LaneTrackState] = {}
        self._next_id = 1

    def update(
        self, raw_id: int, lane_id: int, bbox: np.ndarray, frame_idx: int,
        reserved_ids: set[int] | None = None,
    ) -> int:
        reserved_ids = reserved_ids or set()
        state = self._logical_state.get(self._raw_to_logical.get(raw_id, -1))
        if state is not None and (lane_id < 0 or state.lane_id < 0 or state.lane_id == lane_id):
            logical_id = self._raw_to_logical[raw_id]
        elif raw_id in self._raw_to_logical:
            # A backend ID crossed a configured lane: start a new logical identity.
            if state is not None and frame_idx - state.last_frame <= self.switch_tolerance_frames:
                logical_id = self._raw_to_logical[raw_id]
            else:
                logical_id = self._new_id()
                self._raw_to_logical[raw_id] = logical_id
        else:
            logical_id = self._find_lane_reassociation(lane_id, bbox, frame_idx, reserved_ids)
            if logical_id is None:
                logical_id = self._new_id()
            self._raw_to_logical[raw_id] = logical_id

        if state is None or lane_id >= 0:
            self._logical_state[logical_id] = LaneTrackState(
                lane_id=lane_id,
                last_frame=frame_idx,
                bbox=tuple(float(value) for value in bbox),
            )
        return logical_id

    def _find_lane_reassociation(
        self, lane_id: int, bbox: np.ndarray, frame_idx: int, reserved_ids: set[int],
    ) -> int | None:
        if lane_id < 0:
            return None
        x1, y1, x2, y2 = [float(value) for value in bbox]
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        diagonal = max(float(np.hypot(x2 - x1, y2 - y1)), 1.0)
        candidates = []
        for logical_id, state in self._logical_state.items():
            if logical_id in reserved_ids or state.lane_id != lane_id:
                continue
            if frame_idx - state.last_frame > self.max_gap_frames:
                continue
            sx1, sy1, sx2, sy2 = state.bbox
            state_center = np.array([(sx1 + sx2) / 2, (sy1 + sy2) / 2])
            distance = float(np.linalg.norm(center - state_center))
            if distance <= self.distance_factor * diagonal:
                candidates.append((distance, logical_id))
        return min(candidates)[1] if candidates else None

    def _new_id(self) -> int:
        logical_id = self._next_id
        self._next_id += 1
        return logical_id


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
            return np.zeros(2 * int(np.prod(bins)), dtype=np.float32)
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        y_start, y_end = crop.shape[0] // 10, max(crop.shape[0], 9 * crop.shape[0] // 10)
        x_start, x_end = crop.shape[1] // 10, max(crop.shape[1], 9 * crop.shape[1] // 10)
        center_crop = crop[y_start:y_end, x_start:x_end]
        center_hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        center_hist = cv2.calcHist(
            [center_hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]
        )
        center_hist = cv2.normalize(center_hist, center_hist).flatten()
        embedding = np.concatenate([hist, center_hist]).astype(np.float32)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def mark_lost(self, track_id: int, embedding: np.ndarray, frame_idx: int) -> None:
        self._lost[track_id] = LostTrack(track_id, embedding, frame_idx)

    def discard(self, track_id: int) -> None:
        """Remove um ID da memória de perdidos quando ele volta por movimento."""
        self._lost.pop(track_id, None)

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
            self.byte_track = self._build_byte_track()
            self._lane_byte_tracks: dict[int, sv.ByteTrack] = {}
        self.appearance = AppearanceReID(
            bins=config.reid_fallback.histogram_bins,
            max_lost_age_frames=config.reid_fallback.max_lost_age_frames,
            similarity_threshold=config.reid_fallback.similarity_threshold,
        )
        self.reid = self.appearance if config.reid_fallback.enabled else None
        self.history = TrackHistory()
        self._active_ids_prev_frame: set[int] = set()
        self._embeddings_by_id: dict[int, np.ndarray] = {}
        self.lane_identity = LaneAwareIdentity(
            config.lane_switch_tolerance_frames,
            config.lane_reassociation_max_gap_frames,
            config.lane_reassociation_distance_factor,
        )
        self.persistent_identity = (
            PersistentIdentityManager(config) if config.persistent_identity_enabled else None
        )

    def _build_byte_track(self) -> sv.ByteTrack:
        return sv.ByteTrack(
            track_activation_threshold=self.config.track_activation_threshold,
            lost_track_buffer=self.config.lost_track_buffer,
            minimum_matching_threshold=self.config.minimum_matching_threshold,
            frame_rate=self.config.frame_rate,
        )

    def _update_backend(self, detections: sv.Detections) -> sv.Detections:
        if not (
            self.config.lane_constraint_enabled
            and "lane_id" in detections.data
            and len(detections) > 0
        ):
            return self.byte_track.update_with_detections(detections)

        tracked_by_lane = []
        lane_ids = np.unique(detections.data["lane_id"])
        for lane_id in lane_ids:
            lane_id = int(lane_id)
            lane_detections = detections[detections.data["lane_id"] == lane_id]
            if lane_id not in self._lane_byte_tracks:
                self._lane_byte_tracks[lane_id] = self._build_byte_track()
            tracked_by_lane.append(
                self._lane_byte_tracks[lane_id].update_with_detections(lane_detections)
            )
        return sv.Detections.merge(tracked_by_lane) if tracked_by_lane else sv.Detections.empty()

    def update(self, frame: np.ndarray, detections: sv.Detections, frame_idx: int) -> sv.Detections:
        tracked = self._update_backend(detections)

        if self.reid is not None:
            tracked = self._apply_reid_fallback(frame, tracked, frame_idx)

        if (self.persistent_identity is not None and tracked.tracker_id is not None):
            logical_ids = []
            reserved_ids: set[int] = set()
            for index, (bbox, raw_id) in enumerate(zip(tracked.xyxy, tracked.tracker_id)):
                lane_id = int(tracked.data["lane_id"][index]) if "lane_id" in tracked.data else -1
                logical_id = self.persistent_identity.assign(
                    int(raw_id), lane_id, bbox,
                    self.appearance.compute_embedding(frame, bbox, self.appearance.bins),
                    frame_idx, reserved_ids,
                )
                logical_ids.append(logical_id)
                reserved_ids.add(logical_id)
            tracked.tracker_id = np.asarray(logical_ids, dtype=np.int32)
        elif (self.config.lane_constraint_enabled and "lane_id" in tracked.data
                and tracked.tracker_id is not None):
            logical_ids = []
            reserved_ids: set[int] = set()
            for index, (bbox, raw_id) in enumerate(zip(tracked.xyxy, tracked.tracker_id)):
                lane_id = int(tracked.data["lane_id"][index])
                logical_id = self.lane_identity.update(
                    int(raw_id), lane_id, bbox, frame_idx, reserved_ids,
                )
                logical_ids.append(logical_id)
                reserved_ids.add(logical_id)
            tracked.tracker_id = np.asarray(logical_ids, dtype=np.int32)

        current_ids = set(tracked.tracker_id.tolist()) if tracked.tracker_id is not None else set()

        if self.reid is not None and self.persistent_identity is None:
            for track_id in current_ids:
                self.reid.discard(int(track_id))

        # tracks que sumiram desde o frame anterior -> guardar embedding para possível resgate futuro
        if self.reid is not None and self.persistent_identity is None:
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
                    self._embeddings_by_id[int(tid)] = self.appearance.compute_embedding(
                        frame, bbox, self.appearance.bins,
                    )

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
        occupied_ids = {int(track_id) for track_id in tracked.tracker_id}
        rescued_ids: set[int] = set()
        for i, tid in enumerate(tracked.tracker_id):
            tid = int(tid)
            if tid in self._embeddings_by_id or tid in self._active_ids_prev_frame:
                continue  # já é uma track conhecida, não precisa de resgate
            embedding = self.reid.compute_embedding(frame, tracked.xyxy[i], self.reid.bins)
            rescued_id = self.reid.try_rescue(embedding, frame_idx)
            if rescued_id is not None and rescued_id not in occupied_ids and rescued_id not in rescued_ids:
                new_ids[i] = rescued_id
                rescued_ids.add(rescued_id)
                self._embeddings_by_id.pop(tid, None)
        tracked.tracker_id = new_ids
        return tracked
