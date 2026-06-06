"""RAG 引擎 — 将本地知识库编码为向量，支持语义相似度检索。

工作流程：
  1. 启动时用 TF-IDF 将 40 条知识块转为向量
  2. 存入 FAISS 索引（IndexFlatIP = cosine 相似度）
  3. 查询时：TF-IDF 编码 query → 向量检索 → 按 intent/topic 过滤 → 返回 top-k

对比之前的 build_context_blocks()：
  之前：精确字典查找 f"{topic}::{intent}"，必须 key 完全匹配
  现在：向量检索，意思相近就能召回，不依赖关键词命中

为什么用 TF-IDF 而非 sentence-transformers？
  - sentence-transformers 需要从 HuggingFace 下载模型（80MB）
  - 国内网络常被屏蔽，导致无法下载
  - TF-IDF + FAISS 完全本地运行，零下载，RAG 原理完全一致
  - 如果有网络，把 _VECTORIZER 换成 HuggingFaceEmbeddings 即可升级
"""

import re
from typing import Optional

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from langgraph_practical.knowledge import get_all_documents


class RAGEngine:
    """RAG 检索引擎，管理知识库的向量索引与语义检索。"""

    def __init__(self):
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._index: Optional[faiss.Index] = None
        self._documents: list[dict] = []
        self._ready = False

    # ── 公开接口 ──────────────────────────────────────

    def ensure_index(self) -> None:
        """确保索引已构建。如果未构建则自动建库。"""
        if not self._ready:
            self._build_index()

    def retrieve(
        self,
        query: str,
        k: int = 3,
        intent: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> list[tuple[str, float]]:
        """语义检索最相关的知识块。

        Args:
            query:  用户问题
            k:      返回 top-k 条结果
            intent: 可选，按意图过滤（仅返回该 intent 的文档）
            topic:  可选，按主题过滤（仅返回该 topic 的文档）

        Returns:
            [(文档文本, 相似度分数), ...]  按相似度降序
        """
        self.ensure_index()

        # 编码 query
        q_vec = self._vectorizer.transform([query]).toarray().astype(np.float32)  # type: ignore
        # L2 归一化（使内积 = cosine 相似度）
        q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-10)

        # FAISS 搜索（多取候选用于后续过滤）
        scores, indices = self._index.search(q_vec, k * 5)  # type: ignore

        # 过滤 + 排序
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[idx]
            meta = doc["metadata"]
            if intent is not None and meta["intent"] != intent:
                continue
            if topic is not None and meta["topic"] != topic:
                continue
            results.append((doc["text"], float(score)))
            if len(results) >= k:
                break

        return results

    # ── 内部 ──────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> str:
        """中文 + 英文分词预处理。

        在中文前后加空格，使 TfidfVectorizer 的 token_pattern
        能同时匹配中英文 n-gram。
        """
        # 在中文前后插入空格（让 analyzer='word' 能切到它们）
        text = re.sub(r"([\u4e00-\u9fff])", r" \1 ", text)
        # 合并多余空格
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    def _build_index(self) -> None:
        """加载文档，TF-IDF 编码并建 FAISS 索引。"""
        self._documents = get_all_documents()
        texts = [self._tokenize(d["text"]) for d in self._documents]

        # TF-IDF 向量化（词级 + 中文兼容）
        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            max_features=5000,
            sublinear_tf=True,
        )
        tfidf_matrix = self._vectorizer.fit_transform(texts).toarray()

        # L2 归一化 → 内积 = cosine 相似度
        embeddings = tfidf_matrix.astype(np.float32)
        embeddings = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        )
        dim = embeddings.shape[1]

        # FAISS IndexFlatIP
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        self._ready = True


# ── 全局单例（懒加载） ────────────────────────────────

_ENGINE: Optional[RAGEngine] = None


def get_engine() -> RAGEngine:
    """获取全局 RAG 引擎（首次调用自动建库）。"""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = RAGEngine()
        _ENGINE.ensure_index()
    return _ENGINE


def retrieve(
    query: str,
    k: int = 3,
    intent: Optional[str] = None,
    topic: Optional[str] = None,
) -> list[tuple[str, float]]:
    """快捷调用：全局 RAG 检索。"""
    return get_engine().retrieve(query, k=k, intent=intent, topic=topic)
