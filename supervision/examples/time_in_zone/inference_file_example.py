import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, asdict
from typing import Optional
from inference import get_model
from utils.general import find_in_list, load_zones_config
import supervision as sv
import time

# -------------------- VISUAL SETUP --------------------
COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)

# -------------------- MEDIAPIPE --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# -------------------- THRESHOLDS --------------------
ATTENTION_QUALIFY_SECS     = 10   # attentive seconds before freeze rules apply
INATTENTION_FREEZE_SECS    = 5    # freeze dwell after this long inattentive
ATTENTIVE_CONFIRM_FRAMES   = 3    # consecutive attentive frames to confirm
INATTENTIVE_CONFIRM_FRAMES = 4    # consecutive inattentive frames to confirm
BOX_EMA_ALPHA              = 0.2  # box smoother — lower = smoother

# FIX 1 — Yaw gate: nose offset from eye midpoint / face width.
# ~0 = frontal, ~0.5 = fully side-on. 0.30 ≈ 35° cutoff.
MAX_YAW_ASYMMETRY = 0.30
MIN_EYE_OPENNESS  = 0.008   # lowered so tilted-but-frontal faces pass

# Landmark indices
LEFT_EYE_TB  = [(386, 374), (387, 373), (385, 380)]
RIGHT_EYE_TB = [(159, 145), (158, 153), (160, 144)]
NOSE_TIP     = 1
L_EYE_OUTER  = 33
R_EYE_OUTER  = 263


# -------------------- ATTENTION --------------------
def is_looking_at_screen(crop: np.ndarray) -> bool:
    if crop is None or crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return False
    result = face_mesh.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return False
    lms = max(result.multi_face_landmarks, key=lambda f: len(f.landmark)).landmark
    if len(lms) < 200:
        return False
    # eye openness
    left_open  = float(np.mean([abs(lms[t].y - lms[b].y) for t, b in LEFT_EYE_TB]))
    right_open = float(np.mean([abs(lms[t].y - lms[b].y) for t, b in RIGHT_EYE_TB]))
    if left_open < MIN_EYE_OPENNESS or right_open < MIN_EYE_OPENNESS:
        return False
    # FIX 1 — yaw check: reject turned-away faces
    face_w = abs(lms[R_EYE_OUTER].x - lms[L_EYE_OUTER].x)
    if face_w < 1e-4:
        return False
    yaw = abs(lms[NOSE_TIP].x - (lms[L_EYE_OUTER].x + lms[R_EYE_OUTER].x) / 2) / face_w
    if yaw > MAX_YAW_ASYMMETRY:
        return False
    return True


# -------------------- DB RECORD --------------------
@dataclass
class DwellRecord:
    session_id: int;  tracker_id: int;  zone_index: int
    first_seen_at: float;  last_updated_at: float
    dwell_seconds: float;  attentive_seconds: float
    is_attentive: bool;  is_frozen: bool;  in_grace_period: bool;  qualified: bool

    def to_dict(self): return asdict(self)
    def dwell_mm_ss(self):
        t = int(self.dwell_seconds)
        return f"{t//60:02d}:{t%60:02d}"


# -------------------- PERSON STATE --------------------
class PersonState:
    def __init__(self, tracker_id: int, session_id: int, now: float):
        self.tracker_id = tracker_id
        self.session_id = session_id
        self.first_seen_at = now
        self.dwell_seconds = 0.0
        self._dwell_tick: Optional[float] = now
        self.is_attentive = False
        self.attentive_seconds = 0.0
        self._attn_tick: Optional[float] = None
        self._consec_att = 0
        self._consec_inat = 0
        self._inat_since: Optional[float] = None
        self.is_frozen = False
        self.in_grace_period = False
        self.qualified = False

    def update(self, raw: bool, now: float):
        # debounce
        if raw:
            self._consec_att += 1;  self._consec_inat = 0
        else:
            self._consec_inat += 1; self._consec_att = 0
        if self._consec_att  >= ATTENTIVE_CONFIRM_FRAMES:   self.is_attentive = True
        if self._consec_inat >= INATTENTIVE_CONFIRM_FRAMES: self.is_attentive = False

        if self.is_attentive:
            # accumulate attention
            if self._attn_tick is None: self._attn_tick = now
            self.attentive_seconds += now - self._attn_tick
            self._attn_tick = now
            if self.attentive_seconds >= ATTENTION_QUALIFY_SECS: self.qualified = True
            # only reset inattention clock after confirmed attentive streak
            if self._consec_att >= ATTENTIVE_CONFIRM_FRAMES:
                self._inat_since = None
                self.in_grace_period = False
            # un-freeze if re-qualified
            if self.is_frozen and self.qualified:
                self.is_frozen = False
                self._dwell_tick = now
        else:
            self._attn_tick = None
            if self._consec_inat >= INATTENTIVE_CONFIRM_FRAMES:
                if self._inat_since is None: self._inat_since = now
            if self._inat_since is not None and self.qualified:
                inat_for = now - self._inat_since
                self.in_grace_period = inat_for < INATTENTION_FREEZE_SECS
                if inat_for >= INATTENTION_FREEZE_SECS and not self.is_frozen:
                    self._freeze(now)

        # tick dwell
        if not self.is_frozen and self._dwell_tick is not None:
            self.dwell_seconds += now - self._dwell_tick
            self._dwell_tick = now

    def _freeze(self, now: float):
        freeze_pt = self._inat_since + INATTENTION_FREEZE_SECS
        if self._dwell_tick is not None:
            extra = freeze_pt - self._dwell_tick
            if extra > 0: self.dwell_seconds += extra
            self._dwell_tick = None
        self.is_frozen = True

    def to_record(self, zone_idx: int, now: float) -> DwellRecord:
        return DwellRecord(
            self.session_id, self.tracker_id, zone_idx,
            self.first_seen_at, now,
            round(self.dwell_seconds, 2), round(self.attentive_seconds, 2),
            self.is_attentive, self.is_frozen, self.in_grace_period, self.qualified
        )

    def dwell_mm_ss(self):
        t = int(self.dwell_seconds); return f"{t//60:02d}:{t%60:02d}"

    def box_color(self):
        if self.is_attentive:      return (0, 255, 0)    # green
        if self.in_grace_period:   return (0, 255, 255)  # yellow
        return (0, 0, 255)                                # red


# -------------------- FIX 2 — BOX SMOOTHER --------------------
class BoxSmoother:
    """EMA + jitter gate. Ignores sub-8px wobbles entirely."""
    def __init__(self):
        self._boxes: dict[int, np.ndarray] = {}

    def smooth(self, tid: int, xyxy: np.ndarray) -> np.ndarray:
        if tid not in self._boxes:
            self._boxes[tid] = xyxy.astype(float)
            return self._boxes[tid]
        delta = np.abs(xyxy - self._boxes[tid])
        if np.any(delta > 8):   # only update if box actually moved
            self._boxes[tid] = BOX_EMA_ALPHA * xyxy + (1 - BOX_EMA_ALPHA) * self._boxes[tid]
        return self._boxes[tid]

    def remove(self, tid: int): self._boxes.pop(tid, None)


# -------------------- API HELPERS --------------------
def build_live_payload(zone_states: dict, zone_idx: int, now: float) -> list:
    return [ps.to_record(zone_idx, now).to_dict() for ps in zone_states.values()]

def build_exit_payload(ps: PersonState, zone_idx: int, now: float) -> dict:
    return {**ps.to_record(zone_idx, now).to_dict(), "event": "exit"}


# -------------------- GLOBAL STATE --------------------
session_id_map  = {}
session_counter = [1]
zone_person_states: dict[int, dict[int, PersonState]] = {}
box_smoother = BoxSmoother()

def get_session_id(tid: int) -> int:
    if tid not in session_id_map:
        session_id_map[tid] = session_counter[0]; session_counter[0] += 1
    return session_id_map[tid]


# -------------------- HELPERS --------------------
def is_valid_detection(xyxy, fw: int, fh: int, live: bool) -> bool:
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0: return False
    return (w > fw * 0.03 and h > fh * 0.03) if live else h > fh * 0.10

def is_live_source(source: str) -> bool:
    s = str(source).strip()
    return s.isdigit() or s.startswith(("rtsp://", "http://", "https://"))


# -------------------- MAIN --------------------
def main(
    zone_configuration_path: str,
    source_video_path: str,
    model_id: str = "yolov8n-640",
    confidence_threshold: float = 0.4,
    iou_threshold: float = 0.3,
    classes: list[int] = [],
    roboflow_api_key: str = "",
) -> None:

    live   = is_live_source(source_video_path)
    print(f"Mode: {'LIVE' if live else 'FILE'}")
    model  = get_model(model_id=model_id, api_key=roboflow_api_key)
    source = int(source_video_path) if source_video_path.isdigit() else source_video_path

    if live:
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
        fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        video_info = sv.VideoInfo(width=fw, height=fh, fps=fps, total_frames=None)
    else:
        cap = None
        video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
        fw, fh = video_info.width, video_info.height

    tracker = sv.ByteTrack(
        minimum_matching_threshold=0.8, lost_track_buffer=90,
        track_activation_threshold=0.1, frame_rate=video_info.fps
    )
    polygons = load_zones_config(file_path=zone_configuration_path)
    zones    = [sv.PolygonZone(polygon=p, triggering_anchors=(sv.Position.CENTER,)) for p in polygons]

    for idx in range(len(zones)):
        zone_person_states[idx] = {}
    prev_tids: dict[int, set] = {idx: set() for idx in range(len(zones))}

    cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Processed Video", 1280, 720)

    def process_frame(frame: np.ndarray) -> np.ndarray:
        now = time.time()

        results    = model.infer(frame, confidence=confidence_threshold, iou_threshold=iou_threshold)[0]
        detections = sv.Detections.from_inference(results)
        if classes:
            detections = detections[find_in_list(detections.class_id, classes)]
        valid = np.array([is_valid_detection(xy, fw, fh, live) for xy in detections.xyxy])
        detections = detections[valid]
        detections = tracker.update_with_detections(detections)

        # attention — once per person, outside zone loop
        attn: dict[int, bool] = {}
        for i in range(len(detections)):
            tid = int(detections.tracker_id[i])
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            x1, y1, x2, y2 = max(x1,0), max(y1,0), min(x2,fw), min(y2,fh)
            attn[tid] = is_looking_at_screen(frame[y1:y2, x1:x2])

        out = frame.copy()

        for idx, zone in enumerate(zones):
            out = sv.draw_polygon(scene=out, polygon=zone.polygon, color=COLORS.by_idx(idx))
            in_zone = detections[zone.trigger(detections)]
            cur_tids: set[int] = set()
            labels: list[str] = []

            for i in range(len(in_zone)):
                tid = int(in_zone.tracker_id[i])
                cur_tids.add(tid)
                if tid not in zone_person_states[idx]:
                    zone_person_states[idx][tid] = PersonState(tid, get_session_id(tid), now)
                ps = zone_person_states[idx][tid]
                ps.update(attn.get(tid, False), now)
                labels.append(f"#{ps.session_id} {ps.dwell_mm_ss()}")

            # exits
            for tid in prev_tids[idx] - cur_tids:
                if tid in zone_person_states[idx]:
                    ps = zone_person_states[idx].pop(tid)
                    print(f"[EXIT] Zone {idx} | {build_exit_payload(ps, idx, now)}")
                    box_smoother.remove(tid)
            prev_tids[idx] = cur_tids

            if len(in_zone) == 0: continue

            out = LABEL_ANNOTATOR.annotate(scene=out, detections=in_zone,
                                           labels=labels, custom_color_lookup=np.arange(len(in_zone)))
            for i in range(len(in_zone)):
                tid = int(in_zone.tracker_id[i])
                smooth = box_smoother.smooth(tid, in_zone.xyxy[i])
                x1, y1, x2, y2 = map(int, smooth)
                ps = zone_person_states[idx].get(tid)
                cv2.rectangle(out, (x1,y1), (x2,y2), ps.box_color() if ps else (0,0,255), 2)

        cv2.putText(out, "LIVE" if live else "FILE", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if live else (255,255,0), 2)
        return out

    if live:
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.imshow("Processed Video", cv2.resize(process_frame(frame), (1280,720)))
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        cap.release()
    else:
        for frame in sv.get_video_frames_generator(source_video_path):
            cv2.imshow("Processed Video", cv2.resize(process_frame(frame), (1280,720)))
            if cv2.waitKey(1) & 0xFF == ord("q"): break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings
    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main, as_positional=False)