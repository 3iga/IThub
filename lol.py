import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.ultralytics as fou

EXPORT_DIR = "C:\Users\Zangar Kasengazy\Malshy_attributes\Dataset_Malshy"

classes = ['Sheep', 'Horse', 'Bull', 'Pig', 'Goat', 'Person', 'Dog']

train = foz.load_zoo_dataset(
    "open-images-v7",
    "train",
    label_types=["segmentations"],
    classes = classes,
    max_samples=10000,
    seed=51,
    dataset_name="Farm",
)

train.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    split="train",
    classes=classes,
)