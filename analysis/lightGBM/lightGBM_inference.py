import joblib
import spacy
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

# 加载 spaCy 模型和停用词
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

# 自定义停用词（需与训练时保持一致）
custom_stopwords = set([
    'hour', 'working', 'career', 'job', 'work', 'time', 'position',
    'role', 'experience', 'day', 'team', 'employee', 'people'
])

def preprocess(text):
    """文本预处理，与训练保持一致"""
    if not isinstance(text, str):
        return ""
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.pos_ in ['NOUN', 'ADJ', 'ADV']
        and token.is_alpha
        and not token.is_stop
        and token.lemma_ not in custom_stopwords
    ]
    return " ".join(tokens)

def predict_from_file(
    input_csv_path,
    model_path="trained_lgbm.joblib",
    output_csv_path="inference_results.csv",
    text_column="clean_description"
):
    """加载模型并对新数据进行预测"""
    # 加载模型组件
    bundle = joblib.load(model_path)
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    freq_mask = bundle["freq_mask"]

    # 读取并预处理新数据
    df = pd.read_csv(input_csv_path)
    df["cleaned_desc"] = df[text_column].apply(preprocess)

    # 向量化并应用掩码
    X_all = vectorizer.transform(df["cleaned_desc"])
    X_filtered = X_all[:, freq_mask]

    # 模型预测
    preds = model.predict(X_filtered)
    df["predicted_women_proportion"] = preds

    # 保存预测结果
    df[["clean_description", "predicted_women_proportion"]].to_csv(output_csv_path, index=False)
    print(f"✅ 推理完成，结果已保存至：{output_csv_path}")
