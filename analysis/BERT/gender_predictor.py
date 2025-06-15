# gender_predictor.py
"""
Usage example for gender_predictor:

from gender_predictor import (
    load_data,
    detect_language,
    LightGBMGenderPredictor,
    compute_word_correlations
)

# 1. 加载数据
synthetic, labeled, unlabeled = load_data(
    'synthetic_vacancies_final.csv',
    'labeled_cleaned_full.csv',
    'enhance_cleaned_unlabeled.csv'
)

# 2. 语言检测
labeled = detect_language(labeled)
unlabeled = detect_language(unlabeled)

# 3. 训练模型
predictor = LightGBMGenderPredictor()
predictor.fit(labeled)

# 4. 评估
metrics = predictor.evaluate(labeled)
print(metrics)

# 5. 预测 unlabeled
unlabeled['prediction'] = predictor.predict(unlabeled)

# 6. 词相关性分析
top_words = compute_word_correlations(
    labeled['description'].tolist(),
    labeled['target'].tolist()
)
print(top_words)
"""
import re
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from langdetect import detect
from lightgbm import LGBMRegressor
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from transformers import AutoTokenizer, AutoModel
from scipy.stats import pearsonr


def load_data(
    synthetic_path: str,
    labeled_path: str,
    unlabeled_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载并标准化数据，确保包含必要列。
    """
    def safe_read_csv(path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except Exception as e:
            raise IOError(f"Cannot load {path}: {e}")

    synthetic = safe_read_csv(synthetic_path)
    labeled = safe_read_csv(labeled_path)
    unlabeled = safe_read_csv(unlabeled_path)

    # 保证 labeled 至少有 description, target, language
    if 'language' not in labeled.columns:
        labeled['language'] = 'en'
    labeled = labeled.rename(columns={'jobdescription': 'description'})
    labeled = labeled[['description', 'target', 'language']]

    # 标准化 unlabeled
    unlabeled = unlabeled.rename(columns={
        'positiontitle': 'title',
        'jobdescription': 'description',
        'job_description': 'description'
    })[['title', 'description']]

    return synthetic, labeled, unlabeled


def detect_language(df: pd.DataFrame, text_col: str = 'description') -> pd.DataFrame:
    """
    为 df 添加一列 language，基于 langdetect.detect。
    """
    df = df.copy()
    df['language'] = df[text_col].astype(str).apply(detect)
    return df


def clean_text(text: str) -> str:
    """
    简单清洗：去除 HTML、非字母，统一小写。
    """
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^A-Za-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def extract_lda_features(
    docs: List[str],
    n_topics: int = 10,
    max_features: int = 1000
) -> Tuple[np.ndarray, CountVectorizer, LatentDirichletAllocation]:
    """
    对文档列表进行 LDA 主题抽取，并返回文档-主题分布矩阵。
    """
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    X_counts = vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    X_topics = lda.fit_transform(X_counts)
    return X_topics, vectorizer, lda


def embed_texts(
    docs: List[str],
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 16,
    device: torch.device = None
) -> np.ndarray:
    """
    用预训练 Transformer 模型计算文本嵌入，返回 (n_samples, hidden_size) 的 numpy 数组。
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            toks = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            toks = {k: v.to(device) for k, v in toks.items()}
            out = model(**toks).last_hidden_state[:, 0, :]  # [CLS] token
            embeddings.append(out.cpu().numpy())
    return np.vstack(embeddings)


def train_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict = None
) -> LGBMRegressor:
    """
    训练一个 LightGBM 回归模型。
    """
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'verbosity': -1,
        'random_state': 42
    }
    if params:
        default_params.update(params)
    model = LGBMRegressor(**default_params)
    model.fit(X, y)
    return model


def evaluate_regression(
    model,
    X: np.ndarray,
    y_true: np.ndarray
) -> Dict[str, float]:
    """
    计算 MAE 和 R²。
    """
    y_pred = model.predict(X)
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }


def compute_word_correlations(
    docs: List[str],
    targets: List[float],
    top_k: int = 30
) -> pd.DataFrame:
    """
    计算最相关和最不相关的词与 target 之间的 Pearson 相关系数。
    """
    vec = CountVectorizer(stop_words='english', max_features=2000)
    X = vec.fit_transform(docs)
    words = vec.get_feature_names_out()
    corrs = []
    for i, w in enumerate(words):
        freq = X[:, i].toarray().ravel()
        r, _ = pearsonr(freq, targets)
        corrs.append((w, r))
    df = pd.DataFrame(corrs, columns=['word', 'corr']).sort_values('corr', ascending=False)
    top = pd.concat([df.head(top_k), df.tail(top_k)])
    return top.reset_index(drop=True)


class GenderPredictorBase:
    """
    抽象基类，提供统一的接口。
    """
    def fit(self, df: pd.DataFrame):
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def evaluate(self, df: pd.DataFrame) -> Dict:
        X = df['description']
        y = df['target']
        preds = self.predict(df)
        return evaluate_regression(self.model, self._make_features(df), y)


class LightGBMGenderPredictor(GenderPredictorBase):
    """
    基于 LightGBM 的性别比例预测器。
    """
    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        self.vec = None

    def _make_features(self, df: pd.DataFrame) -> np.ndarray:
        clean_docs = df['description'].apply(clean_text).tolist()
        # 这里仅用词频 + LDA 主题作为示例
        X_count = self.vec.transform(clean_docs).toarray()
        X_lda, _, _ = extract_lda_features(clean_docs, n_topics=10)
        X = np.hstack([X_count, X_lda])
        return self.scaler.transform(X)

    def fit(self, df: pd.DataFrame):
        clean_docs = df['description'].apply(clean_text).tolist()
        y = df['target'].values

        # 构建特征
        self.vec = CountVectorizer(stop_words='english', max_features=1000)
        X_count = self.vec.fit_transform(clean_docs).toarray()
        X_lda, _, _ = extract_lda_features(clean_docs, n_topics=10)
        X = np.hstack([X_count, X_lda])

        # 标准化
        self.scaler = __import__('sklearn').preprocessing.StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)

        # 训练
        self.model = train_lightgbm(X_scaled, y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X_scaled = self._make_features(df)
        return self.model.predict(X_scaled)
