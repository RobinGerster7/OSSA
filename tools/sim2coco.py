import os
import json
import xml.etree.ElementTree as ET
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Convert VOC format annotations to COCO format.")
    parser.add_argument('--ann_dir', help="Directory with VOC annotation XML files.")
    parser.add_argument('--output', help="Output JSON file for COCO formatted annotations.")
    parser.add_argument('--ext', default='xml', help="Extension of the annotation files (default: xml).")
    return parser.parse_args()

def convert_voc_to_coco(ann_dir, output, ext='xml'):
    # Initialize COCO dataset structure
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 26, "name": "car"}]
    }

    # Annotation ID counter
    ann_id = 1

    # List all files in ann_dir with the specified extension
    ann_files = [f[:-len(ext)-1] for f in os.listdir(ann_dir) if f.endswith('.' + ext)]

    for ann_file in ann_files:
        ann_path = os.path.join(ann_dir, f"{ann_file}.{ext}")
        tree = ET.parse(ann_path)
        root = tree.getroot()

        # Use ann_file ID with .jpg for the filename
        image_info = {
            "id": int(ann_file),
            "file_name": f"{ann_file}.jpg",
            "width": int(root.find('size/width').text),
            "height": int(root.find('size/height').text)
        }
        coco_dataset["images"].append(image_info)

        for obj in root.iter('object'):
            label = obj.find('name').text
            if label == "car":
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                width = xmax - xmin
                height = ymax - ymin

                # Construct annotation information
                annotation_info = {
                    "id": ann_id,
                    "image_id": int(ann_file),
                    "category_id": 26,
                    "bbox": [xmin, ymin, width, height],
                    "area": width * height,
                    "iscrowd": 0
                }
                coco_dataset["annotations"].append(annotation_info)
                ann_id += 1

    with open(output, 'w') as outfile:
        json.dump(coco_dataset, outfile)

if __name__ == "__main__":
    args = parse_args()
    convert_voc_to_coco(args.ann_dir, args.output, args.ext)
