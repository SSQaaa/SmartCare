import argparse
from pathlib import Path

from object.trainer import (
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_IMG_SIZE,
    TRAIN_PATIENCE,
    TRAIN_WORKERS,
    normalize_object_name,
    record_object_video,
    run_pipeline,
    train_existing_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="SmartCare personal object trainer.")
    parser.add_argument("name")
    parser.add_argument("--display-name", default="")
    parser.add_argument("--video", default="")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--img", type=int, default=TRAIN_IMG_SIZE)
    parser.add_argument("--workers", type=int, default=TRAIN_WORKERS)
    parser.add_argument("--patience", type=int, default=TRAIN_PATIENCE)
    args = parser.parse_args()

    object_name = normalize_object_name(args.name)
    display_name = args.display_name.strip() or args.name
    if args.train_only:
        train_existing_dataset(
            object_name=object_name,
            display_name=display_name,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img,
            patience=args.patience,
            workers=args.workers,
        )
        return

    video_path = Path(args.video).resolve() if args.video else record_object_video(object_name, args.camera)

    run_pipeline(
        video_path=video_path,
        object_name=object_name,
        display_name=display_name,
        sample_stride=3,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img,
        patience=args.patience,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
