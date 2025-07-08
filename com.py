#데이터 합치기
import pandas as pd

# 경로 설정
orig_path = "/home/park/Desktop/sw중심대학/Data/train.csv"
aug_path = "/home/park/Desktop/sw중심대학/train_aug_10000.csv"  # 증강 파일 (full_text만 있어도 OK)
save_path = "/home/park/Desktop/sw중심대학/train_extended.csv"

# 원본 데이터 로드
df_orig = pd.read_csv(orig_path)

# 증강 데이터 로드
df_aug = pd.read_csv(aug_path)

# 증강 데이터에 필요한 컬럼 추가
df_aug['title'] = "augmented"
df_aug['generated'] = 1

# 컬럼 순서 맞추기
df_aug = df_aug[df_orig.columns]

# 단순 이어붙이기 (append 느낌)
df_extended = pd.concat([df_orig, df_aug], ignore_index=True)

# 저장
df_extended.to_csv(save_path, index=False)
print(f"✅ 총 데이터 수: {len(df_extended)} (원본 {len(df_orig)} + 증강 {len(df_aug)})")
print(f"💾 저장 위치: {save_path}")
