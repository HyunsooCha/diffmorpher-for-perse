import os
import torch
import numpy as np
import cv2
from PIL import Image
from argparse import ArgumentParser
from model import MultiDiffMorpherPipeline
import csv
from natsort import natsorted
from itertools import combinations
from tqdm import tqdm
from natsort import natsorted
import warnings
warnings.filterwarnings("ignore")

# 디렉토리에서 하위 디렉토리 리스트를 구하는 함수
def get_sorted_directories(directory):
    # 하위 디렉토리만 리스트로 추출
    dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    # natsort로 정렬
    sorted_dirs = natsorted(dirs)
    return sorted_dirs

# CSV 파일을 읽고 데이터를 처리하는 함수
def process_csv(file_path):
    processed_data = []
    
    # CSV 파일 열기
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        # 첫 번째 행은 헤더이므로 건너뜀
        next(reader)
        
        # 각 행을 처리
        for row in reader:
            sample_a = row[0]
            sample_b = row[1]
            dir_name = row[2]
            # ratio = float(row[3])  # ratio를 실수로 변환
            
            # 예시로 sample_a와 sample_b에 대한 추가 작업 수행
            # 여기서는 간단히 각 값들을 출력하는 예시를 보여줌
            # print(f"Processing: {sample_a} to {sample_b}, dir_name: {dir_name}, ratio: {ratio}")
            
            # 필요한 처리 후 데이터 저장 (예: 특정 조건을 만족하는 데이터만 필터링)
            processed_data.append({
                "sample_a": sample_a,
                "sample_b": sample_b,
                "dir_name": dir_name,
            })
    return processed_data

# CSV 파일을 생성하는 함수
def create_csv(directory, output_csv):
    sorted_dirs = get_sorted_directories(directory)
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 헤더 작성
        writer.writerow(['sample_a', 'sample_b', 'dir_name'])
        
        # 각 디렉토리 조합을 생성하여 CSV 파일에 작성
        for sample_a, sample_b in combinations(sorted_dirs, 2):
            dir_name = f"{sample_a}_to_{sample_b}"
            writer.writerow([sample_a, sample_b, dir_name])

def main(args, pipeline):
    dataset_list = natsorted([d for d in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, d))])
    print('length of dataset_list:', len(dataset_list))
    dataset_list = dataset_list[args.start_idx:args.end_idx+1]

    for dataset in dataset_list:
        dataset_path = os.path.join(args.dataset_path, dataset)
        print('Processing:', dataset_path)
        pipeline.train_lora_from_images_in_dataset(dataset_path=dataset_path,
                                                   prompt="",
                                                   save_lora_dir=dataset_path,
                                                   lora_epochs=5,
                                                   lora_lr=2e-4,
                                                   lora_rank=16,
                                                   batch_size=8,
                                                   num_inference_steps=50)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="stabilityai/stable-diffusion-2-1-base",
        help="Pretrained model to use (default: %(default)s)"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="",
        help="Path of the dataset (default: %(default)s)"
    )
    parser.add_argument(
        "--use_adain", action="store_true",
        help="Use AdaIN (default: %(default)s)"
    )
    parser.add_argument(
        "--use_reschedule",  action="store_true",
        help="Use reschedule sampling (default: %(default)s)"
    )
    parser.add_argument(
        "--lamb",  type=float, default=0.6,
        help="Lambda for self-attention replacement (default: %(default)s)"
    )
    parser.add_argument(
        "--fix_lora_value", type=float, default=None,
        help="Fix lora value (default: LoRA Interp., not fixed)"
    )
    parser.add_argument(
        "--save_inter", action="store_true",
        help="Save intermediate results (default: %(default)s)"
    )
    parser.add_argument(
        "--num_frames", type=int, default=16,
        help="Number of frames to generate (default: %(default)s)"
    )
    parser.add_argument(
        "--duration", type=int, default=100,
        help="Duration of each frame (default: %(default)s ms)"
    )
    parser.add_argument(
        "--no_lora", action="store_true"
    )
    parser.add_argument(
        "--start_idx", type=int, default=0,
        help="Start index of the dataset (default: %(default)s)"
    )
    parser.add_argument(
        "--end_idx", type=int, default=0,
        help="End index of the dataset (default: %(default)s)"
    )
    args = parser.parse_args()

    # os.makedirs(args.output_path, exist_ok=True)
    pipeline = MultiDiffMorpherPipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float32)
    pipeline.to("cuda")

    main(args, pipeline)