# analysis/data_cleaning.py

import pandas as pd
import re
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
from textblob import TextBlob
from tqdm import tqdm
from itertools import product
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

tqdm.pandas()
DetectorFactory.seed = 42

# ----------------------------
# 清洗函数
# ----------------------------

def light_clean(text):
    """去除 HTML、乱码、冗余信息的轻度清洗"""
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r"(Company info:|Compensation benefits:|Job type:|Certifications:|Remote possible:|location:)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(None\s*)+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"_?font-[a-z\-:; 0-9]+", "", text)
    text = re.sub(r"[\u2028\u2029\xa0]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def moderate_clean(text):
    """去除网页残留词与异常字符的中度清洗"""
    if not isinstance(text, str):
        return ""
    keywords = ['Opslaan', 'Logo', 'Consultancy.nl', 'Consultancy.org']
    for kw in keywords:
        text = text.replace(kw, '')
    text = re.sub(r'\.{3,}', '.', text)
    text = re.sub(r'[!]{3,}', '!', text)
    text = re.sub(r'\?{3,}', '?', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,;:\'\"!?()\[\]€éàèäöü\-–/]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def enhanced_clean(text, lang):
    """增强清洗：结合 moderate 清洗并使用拼写校正（仅限英文）"""
    if not isinstance(text, str):
        return ""
    noise_words = ["Opslaan", "Logo", "Consultancy.nl", "Consultancy.org", "Delen", "Notificatie", "Solliciteer", "Bekijk vacature", "Zoeken Netherlands"]
    for kw in noise_words:
        text = text.replace(kw, '')
    text = re.sub(r'\.{3,}', '.', text)
    text = re.sub(r'[!]{3,}', '!', text)
    text = re.sub(r'\?{3,}', '?', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,;:\'\"!?()\[\]€éàèäöü\-–/]', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    if lang == "en":
        try:
            text = str(TextBlob(text).correct())
        except:
            pass
    return text

# ----------------------------
# 工具函数
# ----------------------------

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def cliffs_delta(x, y):
    n, m = len(x), len(y)
    more = sum(1 for xi, yi in product(x, y) if xi > yi)
    less = sum(1 for xi, yi in product(x, y) if xi < yi)
    return (more - less) / (n * m)

def compute_stats(wp_en, wp_nl):
    t_stat, p_val = ttest_ind(wp_en, wp_nl, equal_var=False)
    u_stat, u_p = mannwhitneyu(wp_en, wp_nl, alternative='two-sided')
    mean_diff = wp_en.mean() - wp_nl.mean()
    pooled_std = np.sqrt((wp_en.std()**2 + wp_nl.std()**2) / 2)
    cohen_d = mean_diff / pooled_std
    delta = cliffs_delta(wp_en.values, wp_nl.values)

    return {
        "t_test": {"t_stat": t_stat, "p_value": p_val},
        "mann_whitney": {"u_stat": u_stat, "p_value": u_p},
        "cohen_d": cohen_d,
        "cliffs_delta": delta
    }