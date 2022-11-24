import fiftyone as fo
from config import IM_HEIGHT as h, IM_WIDTH as w, CLASSES

# A name for the dataset
name = "my-dataset"
# The directory containing the dataset to import
dataset_dir = "data/test/train"
# dataset_dir = "data/test/validation"
# The type of the dataset being imported
dataset_type = fo.types.VOCDetectionDataset 



dataset = fo.Dataset.from_dir(
    data_path=f'{dataset_dir}/images',
    labels_path=f'{dataset_dir}/annotations/xmls',
    dataset_type=dataset_type,
    name=name
)

# View summary info about the dataset
print(dataset)
print(dataset.default_classes)

predictions_view = dataset.view()

with open('predictions.txt', 'r') as f:
      pred_lines = f.readlines()

predictions = {}
for pred in pred_lines:
      key, val = pred.split(',')
      predictions[key] = val

for sample in predictions_view:
    image_name = sample.filepath.split('/')[-1]
    prediction = predictions.get(image_name)
    if prediction is None: continue
    prediction = prediction.strip().split(' ')

    if len(prediction) < 5:
         continue
    
    detections = []
    for i in range(0, len(prediction), 5):
        label = int(prediction[i])
        x1 = int(prediction[i+1])
        y1 = int(prediction[i+2])
        x2 = int(prediction[i+3])
        y2 = int(prediction[i+4])
        rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]

        detections.append(
            fo.Detection(
                label=CLASSES[label],
                bounding_box=rel_box,
                # confidence=1
            )
        )

    # Save predictions to dataset
    sample["predictions"] = fo.Detections(detections=detections)
    sample.save()

# Print the first few samples in the dataset
# print(dataset.head())

session = fo.launch_app(predictions_view)
session.wait()