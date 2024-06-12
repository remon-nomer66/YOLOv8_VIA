from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt

def load_and_predict(image_path, model_path, save_results=True):
    # 学習済みモデルの読み込み
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = YOLO(model_path)
    
    # 画像の読み込み
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 画像で推論
    results = model(img_rgb)

    # 結果の表示
    for result in results:
        result.show()

    # 結果を保存する場合
    if save_results:
        save_dir = os.path.dirname(image_path)
        os.makedirs(save_dir, exist_ok=True)
        for i, result in enumerate(results):
            result.save(os.path.join(save_dir, f"result_{i}.png"))

    return results

if __name__ == '__main__':
    # 推論に使用する画像のパス
    image_path = r'D:\DementiaDepth\Converted\Test0013\smoothing filter\JET\Test0013_JET_191.png'
    
    # 学習済みモデルのパス
    model_path = r'D:\DementiaDepth\yolov8\TEST01_Colormap\results\train3\weights\best.pt'
    
    # 推論の実行
    results = load_and_predict(image_path, model_path)
    print(results)
