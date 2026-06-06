"""PickleSaver — 基于文件的持久化 Checkpointer。

工作原理：
  1. 继承 InMemorySaver，每次 put/put_writes 后自动 pickle 保存到文件
  2. 启动时从文件恢复之前保存的状态
  3. 实现"跨重启恢复"——进程死了，存档还在

对比 InMemorySaver：
  InMemorySaver:        进程一关，所有 checkpoint 消失
  PickleSaver:          重启后从文件恢复，thread 还能继续

对比 SQLite/PostgresSaver：
  那些是生产级方案，需要安装额外驱动
  PickleSaver 只用 Python 内置 pickle，零依赖，教学演示足够
"""

import os
import pickle
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.memory import InMemorySaver


class PickleSaver(InMemorySaver):
    """InMemorySaver + 文件持久化。

    每次 checkpoint 变更后自动 pickle 到磁盘文件，
    启动时自动从文件恢复。
    """

    def __init__(self, file_path: str | Path, **kwargs):
        super().__init__(**kwargs)
        self._file_path = Path(file_path)
        self._restore()

    # ── 公开接口 ──────────────────────────────────────

    @property
    def checkpoint_path(self) -> str:
        """当前存档文件路径。"""
        return str(self._file_path)

    def save(self) -> None:
        """手动触发存档。通常不需要手动调，put/put_writes 会自动保存。"""
        self._persist()

    def clear(self) -> None:
        """清空内存中的 checkpoint（不影响磁盘文件）。"""
        self.storage.clear()
        self.writes.clear()
        self.blobs.clear()

    # ── 重写关键方法 —— 每次写入后自动存档 ──────────

    def put(self, *args, **kwargs):
        result = super().put(*args, **kwargs)
        self._persist()
        return result

    def put_writes(self, *args, **kwargs):
        result = super().put_writes(*args, **kwargs)
        self._persist()
        return result

    # ── 内部：存档与恢复 ──────────────────────────────

    def _persist(self) -> None:
        """将内存中的 storage / writes / blobs 序列化到文件。"""
        data = {
            "storage": dict(self.storage),
            "writes": dict(self.writes),
            "blobs": dict(self.blobs),
        }
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._file_path, "wb") as f:
            pickle.dump(data, f)

    def _restore(self) -> None:
        """从文件恢复之前保存的 checkpoint 状态。"""
        if not self._file_path.exists():
            return
        try:
            with open(self._file_path, "rb") as f:
                data = pickle.load(f)
            self.storage.clear()
            self.writes.clear()
            self.blobs.clear()
            # 恢复 storage
            for thread_id, ns_map in data.get("storage", {}).items():
                self.storage[thread_id] = ns_map
            # 恢复 writes
            for key, writes_map in data.get("writes", {}).items():
                self.writes[key] = writes_map
            # 恢复 blobs
            for key, blob in data.get("blobs", {}).items():
                self.blobs[key] = blob
            count = len(data.get("storage", {}))
            print(f"  ♻️  从存档恢复：{count} 个 thread 的 checkpoint")
        except Exception as e:
            print(f"  ⚠️  存档恢复失败（文件可能损坏）：{e}")


def build_pickle_checkpointer(
    file_path: str | Path = ".checkpoints/graph.pkl",
) -> PickleSaver:
    """快捷创建 PickleSaver。"""
    return PickleSaver(file_path=file_path)
