# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, List, Dict, Set

from .infer import classify as heuristic_classify


class HeuristicClassifier:
    """
    启发式情绪分类器包装类。
    
    封装底层的 classify 函数，并管理关键词配置。
    """
    
    def __init__(self, keywords: Optional[Dict[str, List[str]]] = None):
        """
        初始化分类器。
        
        Args:
            keywords: 情绪关键词字典，如 {"happy": ["开心", ...]}
        """
        self.keywords: Optional[Dict[str, Set[str]]] = None
        if keywords:
            # 转换为 set 以优化查找并匹配 infer.py 的接口要求
            self.keywords = {
                k: set(v) 
                for k, v in keywords.items() 
                if isinstance(v, list)
            }

    def classify(self, text: str, context: Optional[List[str]] = None) -> str:
        """
        对文本进行情绪分类。
        
        Args:
            text: 待分析文本
            context: 上下文列表（可选）
            
        Returns:
            情绪标签 (neutral, happy, sad, angry)
        """
        return heuristic_classify(text, context, self.keywords)
