import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from camera.depth_camera import open_required_depth_camera


ROOT = Path(__file__).resolve().parent
FACE_ROOT = ROOT / "faces"
DEFAULT_MODEL_PATH = ROOT / "face_model.npz"

ESC = 27
Q_KEY = ord("q")
S_KEY = ord("s")

DEFAULT_THRESHOLD = 0.45
THRESHOLD_PERCENTILE = 95
THRESHOLD_MARGIN = 1.10
INSIGHTFACE_MODEL_NAME = "buffalo_s"
DET_SIZE = (256, 256)
RECOGNIZE_EVERY_N_FRAMES = 5
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


def normalize_person_name(name):
    normalized = []
    for ch in name.strip().lower():
        if ch.isalnum():
            normalized.append(ch)
        else:
            normalized.append("_")
    value = "".join(normalized).strip("_")
    while "__" in value:
        value = value.replace("__", "_")
    return value or "person"


def load_face_app(model_name=INSIGHTFACE_MODEL_NAME, det_size=DET_SIZE):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "InsightFace and onnxruntime are required. Install them with: "
            "pip install insightface onnxruntime"
        ) from exc

    app = FaceAnalysis(
        name=model_name,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=det_size)
    return app


def detect_faces(app, frame):
    faces = app.get(frame)
    faces = [face for face in faces if getattr(face, "embedding", None) is not None]
    faces.sort(key=lambda face: face_area(face), reverse=True)
    return faces


def face_area(face):
    x1, y1, x2, y2 = face.bbox.astype(int)
    return max(0, x2 - x1) * max(0, y2 - y1)


def face_box(face, frame_shape):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


def normalize_embedding(embedding):
    vector = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm <= 1e-6:
        return vector
    return vector / norm


def cosine_similarity(vector, matrix):
    return matrix @ vector


def cosine_distance(vector, matrix):
    return 1.0 - cosine_similarity(vector, matrix)


def calibrate_threshold(vectors, labels):
    nearest_same_person = []
    for i, vector in enumerate(vectors):
        same_mask = labels == labels[i]
        same_mask[i] = False
        same_vectors = vectors[same_mask]
        if len(same_vectors) == 0:
            continue
        distances = cosine_distance(vector, same_vectors)
        nearest_same_person.append(float(np.min(distances)))

    if not nearest_same_person:
        return DEFAULT_THRESHOLD

    threshold = float(np.percentile(nearest_same_person, THRESHOLD_PERCENTILE) * THRESHOLD_MARGIN)
    return max(DEFAULT_THRESHOLD, min(0.75, threshold))


def save_person_metadata(person_dir, person_name, display_name, model_name):
    payload = {
        "person_name": person_name,
        "display_name": display_name,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "recognition_backend": "insightface_arcface",
        "model_name": model_name,
    }
    (person_dir / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_person_metadata(person_dir):
    metadata_path = person_dir / "metadata.json"
    if not metadata_path.exists():
        return {
            "person_name": person_dir.name,
            "display_name": person_dir.name,
        }
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def crop_face_preview(frame, face, margin=0.20):
    x1, y1, x2, y2 = face_box(face, frame.shape)
    w = x2 - x1
    h = y2 - y1
    pad_x = int(w * margin)
    pad_y = int(h * margin)
    height, width = frame.shape[:2]
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def save_sample(person_name, sample_dir, saved, frame, face):
    image_path = sample_dir / f"{person_name}_{saved:04d}.jpg"
    embedding_path = sample_dir / f"{person_name}_{saved:04d}.npy"
    preview = crop_face_preview(frame, face)
    if preview is None:
        return False
    cv2.imwrite(str(image_path), preview)
    np.save(str(embedding_path), normalize_embedding(face.embedding))
    return True


def ascii_text(text, fallback="Known"):
    text = str(text)
    return text if text.isascii() else fallback


def collect_samples(person_name, display_name, camera_index, sample_count, interval_frames, model_name):
    display_name = display_name or person_name
    print("Loading InsightFace model for face collection. Please wait...")
    app = load_face_app(model_name=model_name)
    person_dir = FACE_ROOT / person_name
    sample_dir = person_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_person_metadata(person_dir, person_name, display_name, model_name)

    cap = open_required_depth_camera()
    if not cap.isOpened():
        raise RuntimeError("Cannot open Orbbec depth camera.")

    saved = 0
    frame_idx = 0
    print("Only frames with exactly one face will be saved. Press S to save, Q/ESC to finish.")
    try:
        while saved < sample_count:
            ok, frame = cap.read()
            if not ok:
                break

            faces = detect_faces(app, frame)
            single_face = faces[0] if len(faces) == 1 else None
            preview = frame.copy()

            for i, face in enumerate(faces[:5]):
                x1, y1, x2, y2 = face_box(face, frame.shape)
                color = (0, 255, 0) if i == 0 and len(faces) == 1 else (0, 180, 255)
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)

            if len(faces) == 0:
                status = "No face detected. Frame skipped."
            elif len(faces) > 1:
                status = "Multiple faces detected. Frame skipped."
            else:
                status = "One face detected. Auto save enabled."

            line1 = f"Samples: {saved}/{sample_count}"
            line2 = "InsightFace/ArcFace. Only one face is allowed. S save, Q/ESC quit."
            cv2.putText(preview, line1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(preview, line2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            cv2.putText(preview, status, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow("Collect Face Samples", preview)

            key = cv2.waitKey(1) & 0xFF
            should_save = (
                single_face is not None
                and frame_idx % max(1, interval_frames) == 0
            ) or key == S_KEY
            if should_save and single_face is not None:
                if save_sample(person_name, sample_dir, saved, frame, single_face):
                    saved += 1
                    print(f"Saved sample {saved}/{sample_count}: {sample_dir / f'{person_name}_{saved - 1:04d}.jpg'}")
            elif key == S_KEY:
                print(f"Skipped manual save: expected exactly 1 face, got {len(faces)}.")

            if key in (Q_KEY, ESC):
                break
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if saved == 0:
        raise RuntimeError("No face samples were collected.")
    print(f"[OK] Collected {saved} ArcFace samples for {display_name}.")


def iter_sample_embeddings(face_root):
    for person_dir in sorted(face_root.iterdir()):
        sample_dir = person_dir / "samples"
        if not sample_dir.is_dir():
            continue
        metadata = load_person_metadata(person_dir)
        for embedding_path in sorted(sample_dir.glob("*.npy")):
            embedding = np.load(str(embedding_path)).astype(np.float32)
            yield person_dir.name, metadata.get("display_name", person_dir.name), normalize_embedding(embedding)


def train_model(face_root, model_path, model_name=INSIGHTFACE_MODEL_NAME):
    face_root = Path(face_root)
    model_path = Path(model_path)
    vectors = []
    labels = []
    display_names = {}

    if not face_root.exists():
        raise RuntimeError(f"Face data directory does not exist: {face_root}")

    for person_name, display_name, embedding in iter_sample_embeddings(face_root):
        vectors.append(embedding)
        labels.append(person_name)
        display_names[person_name] = display_name

    if not vectors:
        raise RuntimeError("No ArcFace embeddings found. Run collect/enroll again.")

    vectors = np.vstack(vectors).astype(np.float32)
    labels = np.array(labels)
    threshold = calibrate_threshold(vectors, labels)
    metadata = {
        "backend": "insightface_arcface",
        "model_name": model_name,
        "display_names": display_names,
        "threshold": threshold,
        "threshold_note": "Cosine distance threshold. Lower is stricter.",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    tmp_model_path = model_path.with_name(f"{model_path.stem}.tmp{model_path.suffix}")
    np.savez_compressed(
        str(tmp_model_path),
        vectors=vectors,
        labels=labels,
        metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    tmp_model_path.replace(model_path)

    print(f"[OK] Trained ArcFace model with {len(labels)} samples and {len(display_names)} people.")
    print(f"[OK] Recognition threshold: {threshold:.3f}")
    print(f"[OK] Model saved to: {model_path}")


def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise RuntimeError(f"Face model not found: {model_path}")

    print(f"Loading face recognition database: {model_path}")
    with np.load(str(model_path), allow_pickle=False) as data:
        vectors = data["vectors"].astype(np.float32)
        labels = data["labels"].astype(str)
        metadata = json.loads(str(data["metadata"]))
    return vectors, labels, metadata


def predict_embedding(embedding, vectors, labels, threshold):
    vector = normalize_embedding(embedding)
    distances = cosine_distance(vector, vectors)
    best_by_label = {}

    for label in sorted(set(labels)):
        label_distances = np.sort(distances[labels == label])
        best_by_label[label] = float(np.mean(label_distances[: min(3, len(label_distances))]))

    best_label, best_distance = min(best_by_label.items(), key=lambda item: item[1])
    similarity = max(-1.0, min(1.0, 1.0 - best_distance))
    if best_distance > threshold:
        return "unknown", best_distance, similarity

    return best_label, best_distance, similarity


def parse_source(source, camera_index):
    if source:
        try:
            return int(source)
        except ValueError:
            return source
    return camera_index


def recognize(source, camera_index, model_path, threshold, det_size, process_every, frame_width, frame_height, model_name):
    vectors, labels, metadata = load_model(model_path)
    model_name = model_name or metadata.get("model_name", INSIGHTFACE_MODEL_NAME)
    print("Loading InsightFace model for recognition. Please wait...")
    app = load_face_app(model_name=model_name, det_size=(det_size, det_size))
    display_names = metadata.get("display_names", {})
    if threshold is None:
        threshold = float(metadata.get("threshold", DEFAULT_THRESHOLD))
    print(f"Recognition backend: {metadata.get('backend', 'unknown')}")
    print(f"InsightFace model: {model_name}")
    print(f"Recognition threshold: {threshold:.3f}")

    if source:
        parsed_source = parse_source(source, camera_index)
        if isinstance(parsed_source, int):
            raise RuntimeError("Camera input must use the Orbbec depth camera. Remove --source to use the depth camera.")
        cap = cv2.VideoCapture(parsed_source)
    else:
        cap = open_required_depth_camera()
    if not cap.isOpened():
        raise RuntimeError("Cannot open Orbbec depth camera or video file.")
    if frame_width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    if frame_height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_idx = 0
    last_draw_results = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % max(1, process_every) == 0:
                faces = detect_faces(app, frame)
                last_draw_results = []
                for face in faces:
                    x1, y1, x2, y2 = face_box(face, frame.shape)
                    label, distance, similarity = predict_embedding(face.embedding, vectors, labels, threshold)
                    name = display_names.get(label, label) if label != "unknown" else "Unknown"
                    name = ascii_text(name, "Known")
                    last_draw_results.append({
                        "box": (x1, y1, x2, y2),
                        "name": name,
                        "label": label,
                        "distance": distance,
                        "similarity": similarity,
                    })

            for item in last_draw_results:
                x1, y1, x2, y2 = item["box"]
                color = (0, 255, 0) if item["label"] != "unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{item['name']} sim={item['similarity']:.2f}"
                cv2.putText(frame, text, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            cv2.putText(frame, "Face Recognition: Q/ESC to quit", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.imshow("Face Recognition", frame)

            if last_draw_results and frame_idx % 30 == 0:
                print("Faces:", " | ".join(
                    f"{item['name']}(sim={item['similarity']:.3f}, dist={item['distance']:.3f})"
                    for item in last_draw_results[:5]
                ))

            key = cv2.waitKey(1) & 0xFF
            if key in (Q_KEY, ESC):
                break
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def enroll(args):
    collect_samples(
        person_name=args.person_name,
        display_name=args.display_name,
        camera_index=args.camera_index,
        sample_count=args.samples,
        interval_frames=args.interval_frames,
        model_name=args.insightface_model,
    )
    train_model(args.data_dir, args.model, model_name=args.insightface_model)


def build_parser():
    parser = argparse.ArgumentParser(description="SmartCare face recognition.")
    subparsers = parser.add_subparsers(dest="command")

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--person-name", type=str, required=True)
    collect_parser.add_argument("--display-name", type=str, default="")
    collect_parser.add_argument("--camera-index", type=int, default=0)
    collect_parser.add_argument("--samples", type=int, default=20)
    collect_parser.add_argument("--interval-frames", type=int, default=8)
    collect_parser.add_argument("--insightface-model", type=str, default=INSIGHTFACE_MODEL_NAME)
    collect_parser.set_defaults(func=lambda args: collect_samples(
        args.person_name,
        args.display_name or args.person_name,
        args.camera_index,
        max(1, args.samples),
        max(1, args.interval_frames),
        args.insightface_model,
    ))

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-dir", type=Path, default=FACE_ROOT)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    train_parser.add_argument("--insightface-model", type=str, default=INSIGHTFACE_MODEL_NAME)
    train_parser.set_defaults(func=lambda args: train_model(args.data_dir, args.model, args.insightface_model))

    enroll_parser = subparsers.add_parser("enroll")
    enroll_parser.add_argument("--person-name", type=str, required=True)
    enroll_parser.add_argument("--display-name", type=str, default="")
    enroll_parser.add_argument("--camera-index", type=int, default=0)
    enroll_parser.add_argument("--samples", type=int, default=20)
    enroll_parser.add_argument("--interval-frames", type=int, default=8)
    enroll_parser.add_argument("--data-dir", type=Path, default=FACE_ROOT)
    enroll_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    enroll_parser.add_argument("--insightface-model", type=str, default=INSIGHTFACE_MODEL_NAME)
    enroll_parser.set_defaults(func=lambda args: enroll(prepare_person_args(args)))

    recognize_parser = subparsers.add_parser("recognize")
    recognize_parser.add_argument("--source", type=str, default="")
    recognize_parser.add_argument("--camera-index", type=int, default=0)
    recognize_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    recognize_parser.add_argument("--threshold", type=float, default=None)
    recognize_parser.add_argument("--det-size", type=int, default=DET_SIZE[0])
    recognize_parser.add_argument("--process-every", type=int, default=RECOGNIZE_EVERY_N_FRAMES)
    recognize_parser.add_argument("--frame-width", type=int, default=CAMERA_WIDTH)
    recognize_parser.add_argument("--frame-height", type=int, default=CAMERA_HEIGHT)
    recognize_parser.add_argument("--insightface-model", type=str, default="")
    recognize_parser.set_defaults(func=lambda args: recognize(
        args.source,
        args.camera_index,
        args.model,
        args.threshold,
        max(160, args.det_size),
        max(1, args.process_every),
        args.frame_width,
        args.frame_height,
        args.insightface_model,
    ))

    return parser


def prepare_person_args(args):
    args.person_name = normalize_person_name(args.person_name)
    args.display_name = args.display_name.strip() or args.person_name
    args.samples = max(1, args.samples)
    args.interval_frames = max(1, args.interval_frames)
    return args


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    if hasattr(args, "person_name"):
        args = prepare_person_args(args)
    args.func(args)


if __name__ == "__main__":
    main()
