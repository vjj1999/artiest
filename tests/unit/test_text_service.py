"""T10: Text Service MVP 单元测试（不依赖 ASR 模型）"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from services.text.analyzer import TextAnalyzer, TextAnalyzerConfig, _count_tokens, _detect_cta


class TestTokenCount:
    """验证 token 统计"""

    def test_纯中文(self):
        assert _count_tokens("今天天气不错") == 6

    def test_纯英文(self):
        assert _count_tokens("hello world") == 2

    def test_中英混合(self):
        assert _count_tokens("今天用Python写代码") == 7  # 6中 + 1英

    def test_空字符串(self):
        assert _count_tokens("") == 0

    def test_含标点(self):
        assert _count_tokens("你好，世界！") == 4


class TestCTADetection:
    """验证 CTA 关键词检测"""

    def test_检测到关键词(self):
        keywords = ["点击", "关注", "订阅"]
        matches = _detect_cta("请点击关注并订阅", keywords)
        assert len(matches) == 3

    def test_无关键词(self):
        matches = _detect_cta("今天天气很好", ["点击", "关注"])
        assert len(matches) == 0

    def test_重复关键词(self):
        matches = _detect_cta("点击这里点击那里", ["点击"])
        assert len(matches) == 1
        assert matches[0]["count"] == 2

    def test_位置正确(self):
        matches = _detect_cta("请点击", ["点击"])
        assert matches[0]["positions"] == [1]


class TestTextAnalyzer:
    """验证 TextAnalyzer 文本分析（跳过 ASR）"""

    def test_analyze_text基础(self):
        analyzer = TextAnalyzer(TextAnalyzerConfig(disable_asr=True))
        result = analyzer.analyze_text("今天天气很好点击关注", duration_s=10.0)
        assert result["token_count"] == 10
        assert result["token_per_sec"] == 1.0
        assert result["has_cta"] is True
        assert result["duration_s"] == 10.0

    def test_语速计算(self):
        analyzer = TextAnalyzer(TextAnalyzerConfig(disable_asr=True))
        result = analyzer.analyze_text("一二三四五六七八九十", duration_s=5.0)
        assert result["token_per_sec"] == 2.0

    def test_无CTA(self):
        analyzer = TextAnalyzer(TextAnalyzerConfig(disable_asr=True))
        result = analyzer.analyze_text("今天讨论了技术问题", duration_s=3.0)
        assert result["has_cta"] is False
        assert result["cta_matches"] == []

    def test_空文本(self):
        analyzer = TextAnalyzer(TextAnalyzerConfig(disable_asr=True))
        result = analyzer.analyze_text("", duration_s=0.0)
        assert result["token_count"] == 0
        assert result["token_per_sec"] == 0.0

    def test_输出字段完整(self):
        analyzer = TextAnalyzer(TextAnalyzerConfig(disable_asr=True))
        result = analyzer.analyze_text("测试", duration_s=1.0)
        for key in ["transcript", "token_count", "token_per_sec", "duration_s", "cta_matches", "has_cta"]:
            assert key in result
