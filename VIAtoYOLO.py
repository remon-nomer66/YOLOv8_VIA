import os
import shutil
import glob
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import yaml

# CSVファイルの絶対パスを指定
csv_path = r'D:\DementiaDepth\yolov8\TEST01_Colormap\VIA_TEST_02.csv'

# CSVファイルの読み込み
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

df = pd.read_csv(csv_path)

# デバッグ用に最初の数行を表示
print(df.head())

# 必要な情報を抽出し、新しいDataFrameを作成
annotations = []
for _, row in df.iterrows():
    shape_attr = json.loads(row['region_shape_attributes'])
    print(f"shape_attr: {shape_attr}")  # デバッグ出力
    if 'x' in shape_attr and 'y' in shape_attr and 'width' in shape_attr and 'height' in shape_attr:
        annotations.append({
            'filename': row['filename'],
            'xmin': shape_attr['x'],
            'ymin': shape_attr['y'],
            'xmax': shape_attr['x'] + shape_attr['width'],
            'ymax': shape_attr['y'] + shape_attr['height']
        })
    else:
        print(f"Missing keys in shape_attr: {shape_attr}")  # デバッグ出力

annotations_df = pd.DataFrame(annotations)

# ディレクトリの設定
BASE_DIR = r'D:\DementiaDepth\yolov8\TEST01_Colormap'
DATASET_DIR = os.path.join(BASE_DIR, "datasets/person")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
# TEST_DIR = os.path.join(BASE_DIR, "test_images")  # 必要に応じて使用

# 訓練用と評価用のフォルダを作成
os.makedirs(os.path.join(IMAGES_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(LABELS_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(IMAGES_DIR, "valid"), exist_ok=True)
os.makedirs(os.path.join(LABELS_DIR, "valid"), exist_ok=True)

# 画像ファイルのリストを取得
image_files = glob.glob(os.path.join(TRAIN_DIR, "*.png"))

# デバッグ用にファイルリストを表示
print(f"Image files found: {image_files}")

if not image_files:
    raise FileNotFoundError(f"No image files found in the directory: {TRAIN_DIR}")

# 訓練用と評価用に分割（8:2の割合）
train_files, valid_files = train_test_split(image_files, test_size=0.2)

# ファイルをそれぞれのフォルダにコピー
for file in train_files:
    shutil.copy(file, os.path.join(IMAGES_DIR, "train"))

for file in valid_files:
    shutil.copy(file, os.path.join(IMAGES_DIR, "valid"))

# アノテーションデータをYOLOフォーマットに変換
for _, row in annotations_df.iterrows():
    image_file = row['filename']
    class_id = "0"  # クラスIDを設定（peopleとして0としています）
    x_min = row['xmin']
    y_min = row['ymin']
    x_max = row['xmax']
    y_max = row['ymax']

    # YOLOフォーマットに変換
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    # 正規化
    x_center /= 320
    y_center /= 180
    width /= 320
    height /= 180

    # アノテーションファイルのパスを決定
    if os.path.join(TRAIN_DIR, image_file) in train_files:
        annotation_file = os.path.join(LABELS_DIR, "train", image_file.replace('.png', '.txt'))
    else:
        annotation_file = os.path.join(LABELS_DIR, "valid", image_file.replace('.png', '.txt'))

    # アノテーションデータをファイルに書き込み
    with open(annotation_file, 'a') as ann_file:
        ann_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# yamlファイルの生成
yaml_content = {
    'path': DATASET_DIR,
    'train': os.path.join(IMAGES_DIR, "train"),
    'val': os.path.join(IMAGES_DIR, "valid"),
    'nc': 1,  # クラス数（ここではpeopleのみ）
    'names': ['person']  # クラス名
}

yaml_path = os.path.join(BASE_DIR, 'dataset.yaml')
with open(yaml_path, 'w') as yaml_file:
    yaml.dump(yaml_content, yaml_file, default_flow_style=False)

print(f"yaml file created at: {yaml_path}")
