import os
import shutil

from database import SessionLocal
from models import Prediction
from class_mapping import CLASS_MAPPING


RETRAINING_FOLDER = "uploads/retraining"


def build_retraining_dataset():

    db = SessionLocal()

    candidates = (
        db.query(Prediction)
        .filter(
            Prediction.is_valid == True,
            Prediction.used_for_retraining == False
        )
        .all()
    )

    print(f"Found {len(candidates)} candidate images")

    os.makedirs(
        RETRAINING_FOLDER,
        exist_ok=True
    )

    copied_count = 0

    for item in candidates:

        source_path = item.processed_image_path

        class_name = CLASS_MAPPING[
            item.predicted_class
        ]

        class_folder = os.path.join(
            RETRAINING_FOLDER,
            class_name
        )

        os.makedirs(
            class_folder,
            exist_ok=True
        )

        if os.path.exists(source_path):

            destination_path = os.path.join(
                class_folder,
                os.path.basename(source_path)
            )

            shutil.copy2(
                source_path,
                destination_path
            )

            copied_count += 1

    db.close()

    print(
        f"Dataset build completed. "
        f"{copied_count} images copied."
    )