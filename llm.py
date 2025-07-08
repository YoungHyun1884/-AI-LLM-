#AI로 생성한 문장 만들기
from transformers import AutoTokenizer, BartForConditionalGeneration, pipeline
import pandas as pd

# === 설정 ===
MODEL_NAME = "guialfaro/korean-paraphrasing"
TRAIN_PATH = "/home/park/Desktop/sw중심대학/Data/train.csv"
SAVE_PATH = "/home/park/Desktop/sw중심대학/train_aug_1000.csv"
NUM_SAMPLES = 10000  # 증강할 문장 수 (예: 1000)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
paraphraser = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)


def local_paraphrase(text):
    try:
        input_text = f"paraphrase: {text[:500]}"  # 너무 긴 문장은 잘라서 입력
        result = paraphraser(
            input_text,
            max_new_tokens=100,
            num_beams=2,
            no_repeat_ngram_size=2,
            do_sample=False
        )
        return result[0]['generated_text']
    except Exception as e:
        print("❌ 오류 발생:", e)
        return text


df = pd.read_csv(TRAIN_PATH)
df_1 = df[df['generated'] == 1]

n_sample = min(NUM_SAMPLES, len(df_1))
print(f"✅ 사용할 샘플 수: {n_sample} / 전체 available: {len(df_1)}")

ai_df = df_1.sample(n=n_sample, random_state=42).copy()


ai_df['full_text_original'] = ai_df['full_text']
ai_df['full_text'] = ai_df['full_text'].apply(local_paraphrase)

ai_df.to_csv(SAVE_PATH, index=False)
print(f"✅ 증강 완료 및 저장: {SAVE_PATH}")
