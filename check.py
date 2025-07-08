#인공지능으로 생성 된 문장 확인하기
import pandas as pd

df = pd.read_csv("/home/park/Desktop/sw중심대학/train_aug_10000.csv")
num_total = len(df)
num_same = (df["full_text"] == df["full_text_original"]).sum()
num_changed = num_total - num_same

print(f"총 샘플 수: {num_total}")
print(f"성공적으로 증강된 문장 수: {num_changed}")
print(f"실패(원문 유지)된 문장 수: {num_same}")
