import pandas as pd
import numpy as np
import spacy
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
import os

# 加载 spaCy 模型和停用词
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

# 可自定义扩展的领域停用词
custom_stopwords = set([
    'hour', 'working', 'career', 'job', 'work', 'time', 'position',
    'role', 'experience', 'day', 'team', 'employee', 'people'
])

def preprocess(text):
    """文本预处理（小写、词形还原、去停用词与自定义词）"""
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

def train_lgbm_model(
    csv_path,
    output_model_path="trained_lgbm.joblib",
    text_column="clean_description",
    target_column="women_proportion"
):
    """训练 LGBM 模型并保存（包括 TF-IDF 向量器与掩码）"""
    df = pd.read_csv(csv_path)
    df["cleaned_desc"] = df[text_column].apply(preprocess)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["cleaned_desc"])

    # 筛掉太常见/太罕见的词项（文档频率过滤）
    doc_freq = (X > 0).sum(axis=0).A1 / X.shape[0]
    freq_mask = (doc_freq > 0.05) & (doc_freq < 0.7)
    X_filtered = X[:, freq_mask]

    y = df[target_column].values

    # 拆分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y, test_size=0.2, random_state=42
    )

    model = LGBMRegressor()
    model.fit(X_train, y_train)

    # 保存模型、向量器、掩码
    joblib.dump({
        "model": model,
        "vectorizer": vectorizer,
        "freq_mask": freq_mask
    }, output_model_path)

    print(f"✅ 模型训练完成，已保存至：{output_model_path}")
    return model