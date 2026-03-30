import os
import subprocess
import re
import yaml # PyYAML 라이브러리

EASYVOLCAP_DATA_ROOT = "data/datasets/ref-dl3dv"
CONFIG_OUTPUT_DIR = "scripts_formatting" # 생성된 yaml 파일이 저장될 경로
METADATA_SCRIPT_PATH = "scripts/preprocess/tools/compute_metadata.py"

# 생성될 YAML 파일의 기본 템플릿
YAML_TEMPLATE = """
# Auto-generated config for scene: {scene_name}
configs: configs/datasets/ref-dl3dv/{scene_name}.yaml

dataloader_cfg:
    dataset_cfg: &dataset_cfg
        ratio: 0.5
        data_root: {data_root}
        view_sample: {train_view_sample}

val_dataloader_cfg:
    dataset_cfg:
        <<: *dataset_cfg
        view_sample: {val_view_sample}

model_cfg:
    sampler_cfg:
        preload_gs: {preload_gs_path}
        spatial_scale: {spatial_scale}
        # Environment Gaussian
        env_preload_gs: {env_preload_gs_path}
        env_bounds: {env_bounds}
"""

def extract_metadata(scene_name):
    """지정된 scene에 대해 metadata 추출."""
    print(f"[*] Extracting metadata for scene: {scene_name}")
    command = [
        "python",
        METADATA_SCRIPT_PATH,
        "--data_root", EASYVOLCAP_DATA_ROOT,
        "--scenes", scene_name,
        "--eval"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[!] Error running metadata script for {scene_name}:")
        print(e.stderr)
        return None

def parse_metadata_output(output):
    metadata = {}
    try:
        # 정규표현식을 사용하여 각 필드를 찾습니다. re.DOTALL은 줄바꿈을 포함하여 매칭합니다.
        metadata['train_view_sample'] = re.search(r"dataloader_cfg\.dataset_cfg\.view_sample: (\[.*?\])", output, re.DOTALL).group(1)
        metadata['val_view_sample'] = re.search(r"val_dataloader_cfg\.dataset_cfg\.view_sample: (\[.*?\])", output, re.DOTALL).group(1)
        metadata['spatial_scale'] = re.search(r"model_cfg\.sampler_cfg\.spatial_scale: ([\d\.]+)", output).group(1)
        metadata['env_bounds'] = re.search(r"model_cfg\.sampler_cfg\.env_bounds: (\[\[.*?\]\])", output, re.DOTALL).group(1)
        return metadata
    except AttributeError as e:
        print(f"[!] Failed to parse metadata. A value might be missing in the output.")
        print(f"    Error: {e}")
        return None

def main():
    if not os.path.isdir(EASYVOLCAP_DATA_ROOT):
        print(f"[!] Error: Data root directory not found at '{EASYVOLCAP_DATA_ROOT}'")
        return

    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
    
    scenes = [d for d in os.listdir(EASYVOLCAP_DATA_ROOT) if os.path.isdir(os.path.join(EASYVOLCAP_DATA_ROOT, d))]

    for scene_name in scenes:
        metadata_output = extract_metadata(scene_name)
        if not metadata_output:
            continue

        parsed_data = parse_metadata_output(metadata_output)
        if not parsed_data:
            continue
            
        print(f"[*] Successfully parsed metadata for {scene_name}")

        # YAML 파일 생성을 위한 경로 및 변수 설정
        scene_data_root = os.path.join(EASYVOLCAP_DATA_ROOT, scene_name)
        
        # 템플릿에 값 채우기
        yaml_content = YAML_TEMPLATE.format(
            scene_name=scene_name,
            data_root=scene_data_root,
            train_view_sample=parsed_data['train_view_sample'],
            val_view_sample=parsed_data['val_view_sample'],
            preload_gs_path=os.path.join(scene_data_root, "sparse/0/points3D.ply"),
            spatial_scale=parsed_data['spatial_scale'],
            env_preload_gs_path=os.path.join(scene_data_root, "envs/points3D.ply"),
            env_bounds=parsed_data['env_bounds']
        )
        
        # YAML 파일로 저장
        output_yaml_path = os.path.join(CONFIG_OUTPUT_DIR, f"{scene_name}.yaml")
        with open(output_yaml_path, 'w') as f:
            # 문자열을 그대로 파일에 씁니다.
            f.write(yaml_content.strip())
            
        print(f"[*] Successfully generated config file: {output_yaml_path}\n")

    print("All config files have been generated.")

if __name__ == "__main__":
    main()