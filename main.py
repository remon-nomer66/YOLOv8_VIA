from ultralytics import YOLO
import os

def train_model(yaml_path, repeat_count=3, epochs=5, batch_size=8):
    # yaml ファイルの存在確認
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    base_save_dir = os.path.join(os.path.dirname(yaml_path), 'results')
    os.makedirs(base_save_dir, exist_ok=True)
    
    precision_list = []
    recall_list = []
    map50_list = []
    map50_95_list = []
    preprocess_speed_list = []
    inference_speed_list = []
    loss_speed_list = []
    postprocess_speed_list = []

    for i in range(repeat_count):
        save_dir = os.path.join(base_save_dir, f'result_{i+1}')
        os.makedirs(save_dir, exist_ok=True)

        try:
            # モデルのトレーニング
            model = YOLO('yolov8n.pt')
            model.train(data=yaml_path, epochs=epochs, batch=batch_size, device='0', project=save_dir)
            
            # モデルの評価
            results = model.val()
            print(results)
            
            # 結果をリストに追加
            precision_list.append(results.results_dict['metrics/precision(B)'])
            recall_list.append(results.results_dict['metrics/recall(B)'])
            map50_list.append(results.results_dict['metrics/mAP50(B)'])
            map50_95_list.append(results.results_dict['metrics/mAP50-95(B)'])
            preprocess_speed_list.append(results.speed['preprocess'])
            inference_speed_list.append(results.speed['inference'])
            loss_speed_list.append(results.speed['loss'])
            postprocess_speed_list.append(results.speed['postprocess'])
            
            # 結果をテキストファイルに保存
            result_text = (
                f"Precision: {results.results_dict['metrics/precision(B)']}\n"
                f"Recall: {results.results_dict['metrics/recall(B)']}\n"
                f"mAP50: {results.results_dict['metrics/mAP50(B)']}\n"
                f"mAP50-95: {results.results_dict['metrics/mAP50-95(B)']}\n"
                f"Fitness: {results.results_dict['fitness']}\n"
                f"Speed (preprocess): {results.speed['preprocess']}\n"
                f"Speed (inference): {results.speed['inference']}\n"
                f"Speed (loss): {results.speed['loss']}\n"
                f"Speed (postprocess): {results.speed['postprocess']}\n"
            )
            
            result_file_path = os.path.join(save_dir, 'evaluation_results.txt')
            with open(result_file_path, 'w') as f:
                f.write(result_text)
            
            print(f"Results saved to: {result_file_path}")

        except Exception as e:
            print(f"Error during training or evaluation: {e}")
    
    # 精度とスピードの平均値を計算
    avg_precision = sum(precision_list) / repeat_count
    avg_recall = sum(recall_list) / repeat_count
    avg_map50 = sum(map50_list) / repeat_count
    avg_map50_95 = sum(map50_95_list) / repeat_count
    avg_preprocess_speed = sum(preprocess_speed_list) / repeat_count
    avg_inference_speed = sum(inference_speed_list) / repeat_count
    avg_loss_speed = sum(loss_speed_list) / repeat_count
    avg_postprocess_speed = sum(postprocess_speed_list) / repeat_count

    avg_results_text = (
        f"Average Precision: {avg_precision}\n"
        f"Average Recall: {avg_recall}\n"
        f"Average mAP50: {avg_map50}\n"
        f"Average mAP50-95: {avg_map50_95}\n"
        f"Average Speed (preprocess): {avg_preprocess_speed}\n"
        f"Average Speed (inference): {avg_inference_speed}\n"
        f"Average Speed (loss): {avg_loss_speed}\n"
        f"Average Speed (postprocess): {avg_postprocess_speed}\n"
    )

    avg_results_file_path = os.path.join(base_save_dir, 'average_evaluation_results.txt')
    with open(avg_results_file_path, 'w') as f:
        f.write(avg_results_text)

    print(f"Average results saved to: {avg_results_file_path}")

if __name__ == '__main__':
    # 1つのYAMLファイルを指定
    yaml_path = r'D:\DementiaDepth\yolov8\TEST01_Colormap\dataset.yaml'
    
    print(f"Training with dataset: {yaml_path}")
    train_model(yaml_path, repeat_count=3, epochs=5, batch_size=8)
