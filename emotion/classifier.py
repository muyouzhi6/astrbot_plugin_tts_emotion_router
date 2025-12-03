# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, List

from .infer import classify as heuristic_classify


class HeuristicClassifier:
    def __init__(self):
        pass

    def classify(self, text: str, context: Optional[List[str]] = None) -> str:
        """简单启发式分类，必要时可替换为更强模型。"""
        return heuristic_classify(text, context)
