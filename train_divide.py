import pandas as pd

# 경로 설정
train_path = "/home/park/Desktop/sw중심대학/Data/train.csv"
output_path = "/home/park/Desktop/sw중심대학/Data/train_paragraphs.csv"

# 결과 저장할 리스트
paragraphs = []

# chunk 단위로 나눠서 메모리 절약
chunk_size = 500  # 필요 시 조절 가능

for chunk in pd.read_csv(train_path, chunksize=chunk_size):
    for idx, row in chunk.iterrows():
        title = row["title"]
        full_text = row["full_text"]
        label = row["generated"]

        # 문단 분리 (빈 줄 기준, '\n\n')
        split_paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]

        for i, para in enumerate(split_paragraphs):
            paragraphs.append({
                "title": title,
                "paragraph_index": i,
                "paragraph_text": para,
                "generated": label
            })

# DataFrame 생성
paragraph_df = pd.DataFrame(paragraphs)

# CSV로 저장
paragraph_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"문단 단위로 나눈 학습 데이터를 다음 경로에 저장했습니다:\n{output_path}")
