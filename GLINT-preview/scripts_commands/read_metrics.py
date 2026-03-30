import os
import json
import pandas as pd

# 1. 결과가 들어있는 상위 디렉토리 경로
ROOT_DIR = 'data/result/envgs/ref-dl3dv/0807_use_spec_residual'

# 2. 원하는 순서의 4자리 프리픽스 리스트
order_prefixes = [
    '6b42','10dc','194d','543b','3712',
    '5454','5241','a19e','b9df','b65e',
    'b626','eb65'
]

# 3. 결과를 담을 리스트
records = []

# 4. 각 prefix별로 디렉토리 확인
for prefix in order_prefixes:
    # 이 프리픽스로 시작하는 실제 디렉토리명을 찾는다
    matches = [d for d in os.listdir(ROOT_DIR) if d.startswith(prefix)]
    for scene_name in sorted(matches):  # 혹시 동일 prefix 아래 여러 디렉토리가 있으면 알파벳순
        scene_dir = os.path.join(ROOT_DIR, scene_name)
        json_path = os.path.join(scene_dir, 'metrics.json')
        if not os.path.isfile(json_path):
            continue

        # metrics.json 읽어서 summary 추출
        with open(json_path, 'r') as f:
            data = json.load(f)
        summary = data.get('summary', {})

        # 관심 값만 기록
        records.append({
            'scene': scene_name,
            'psnr_mean': summary.get('psnr_mean'),
            'ssim_mean': summary.get('ssim_mean'),
            'lpips_mean': summary.get('lpips_mean'),
        })

# 5. pandas DataFrame으로 변환
df = pd.DataFrame(records)

# 6. Markdown 테이블로 출력
print(df.to_markdown(index=False))

# -- 옵션: CSV로 저장하고 싶다면 아래 주석을 해제하세요 --
# df.to_csv('all_scenes_metrics_custom_order.csv', index=False)
