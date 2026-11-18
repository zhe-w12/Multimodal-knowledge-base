"""Deterministic intent, query rewriting and subject-document NER helpers."""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, Iterable, List

from manufacturing_rag.schemas import EntityExtractionResult, IntentResult


LEARNING_TERMS = (
    "文档",
    "资料",
    "课程",
    "课堂",
    "教材",
    "章节",
    "知识点",
    "概念",
    "例题",
    "讲义",
    "课件",
    "学习目标",
    "教学目标",
    "定义",
    "定理",
    "公式",
    "证明",
    "原文",
    "材料",
    "思想",
    "理论",
)

ASSIGNMENT_TERMS = (
    "作业",
    "习题",
    "练习",
    "任务",
    "提交",
    "评分标准",
    "答案解析",
    "批改",
    "项目作业",
)

EXAM_TERMS = (
    "考试",
    "测验",
    "试卷",
    "真题",
    "复习",
    "考点",
    "题型",
    "分值",
    "模拟题",
    "备考",
)

POLICY_TERMS = (
    "教务",
    "选课",
    "学分",
    "培养方案",
    "课程大纲",
    "请假",
    "补考",
    "重修",
    "成绩",
    "考勤",
)

MALICIOUS_TERMS = (
    "忽略以上",
    "忽略前文",
    "绕过",
    "越权",
    "提示注入",
    "系统提示词",
    "developer message",
    "删除所有",
    "清空数据库",
    "shutdown",
)

UNAUTHORIZED_TERMS = (
    "导出全部",
    "批量下载",
    "未授权",
    "绝密",
    "机密",
    "商业秘密",
    "内部题库",
    "泄题",
    "学生隐私",
    "成绩单",
    "密码",
    "密钥",
    "api_key",
    "token",
)

LEARNING_NODES = (
    "课前预习",
    "课堂讲解",
    "例题演示",
    "随堂练习",
    "课后作业",
    "单元复习",
    "实验实践",
    "项目探究",
    "阶段测验",
    "期末复习",
)

STOP_TERMS = {"这个", "资料", "里面", "里的", "是多少", "什么", "需要", "哪些", "如何", "怎么", "是否", "确认", "可以", "一下"}

# Small deterministic synonym expansions keep offline mode useful and provide a
# safe baseline for the production query agent.  Expansions add retrieval
# vocabulary only; they never invent a concrete product, number, or answer.
QUERY_EXPANSIONS = (
    ("退货", "售后申请 退换货"),
    ("退款", "售后申请 退款"),
    ("截图", "操作方法 屏幕截图"),
    ("怎么学", "学习方法 学习目标 知识点"),
    ("如何复习", "复习 考点 知识点 题型"),
    ("常见错误", "易错点 注意事项 错误分析"),
    ("为什么", "原理 原因 机制"),
    ("定义", "概念 定义 解释"),
    ("证明", "推导 证明 数学原理"),
    ("作业答案", "习题 作业 答案解析"),
)


def normalize_text(text: str) -> str:
    normalized = (text or "").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.replace("，", ",").replace("。", ".").replace("：", ":")
    return normalized


def unique_keep_order(values: Iterable[str]) -> List[str]:
    seen = OrderedDict()
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen[cleaned] = None
    return list(seen.keys())


def split_terms(text: str) -> List[str]:
    """Extract stable retrieval terms without external tokenizers."""

    normalized = normalize_text(text)
    lower = normalized.lower()
    terms: List[str] = []
    terms.extend(re.findall(r"\d+[a-zA-Z][a-zA-Z0-9_-]*|[a-zA-Z]+[-_]?[a-zA-Z0-9]*|\d+(?:\.\d+)?(?:分|页|课时|学分|%)?", lower))
    for term in LEARNING_TERMS + ASSIGNMENT_TERMS + EXAM_TERMS + POLICY_TERMS + LEARNING_NODES:
        if term.lower() in lower:
            terms.append(term.lower())
    chinese_only = re.sub(r"[^\u4e00-\u9fa5]", "", normalized)
    for size in (2, 3, 4):
        for index in range(0, max(len(chinese_only) - size + 1, 0)):
            gram = chinese_only[index : index + size]
            if len(gram) == size and gram not in STOP_TERMS:
                terms.append(gram)
    return unique_keep_order(terms)


class RuleBasedQueryProcessor:
    """Small deterministic substitute for LLM routing prompts in local MVP mode."""

    course_pattern = re.compile(r"(?:课程|科目|学科|课程号|课号)\s*(?:编号|代码|ID|号)?\s*[:：#-]?\s*([A-Za-z0-9\u4e00-\u9fa5][A-Za-z0-9\u4e00-\u9fa5\-_/]{1,31})", re.I)
    grade_pattern = re.compile(r"([一二三四五六七八九十高初小本硕博0-9]{1,4}\s*(?:年级|班|学期))")
    knowledge_pattern = re.compile(r"(?:知识点|考点|概念|主题)\s*[:：#-]?\s*([A-Za-z0-9\u4e00-\u9fa5][A-Za-z0-9\u4e00-\u9fa5\-_/]{1,31})", re.I)
    exam_pattern = re.compile(r"(期中|期末|月考|单元测验|模拟考|随堂测验|中考|高考|考研|四六级)")
    assignment_pattern = re.compile(r"(?:作业|任务|练习)\s*(?:编号|ID|号)?\s*[:：#-]?\s*([A-Za-z0-9][A-Za-z0-9\-_/]{0,31})", re.I)
    score_pattern = re.compile(r"\d+(?:\.\d+)?\s*(?:分|学分|课时|%)", re.I)

    def classify_intent(self, query: str) -> IntentResult:
        normalized = normalize_text(query)
        lower = normalized.lower()
        if not normalized:
            return IntentResult("empty", False, 1.0, "空问题无法进入检索链路")
        if any(term.lower() in lower for term in MALICIOUS_TERMS):
            return IntentResult("malicious_attack", False, 0.96, "命中提示注入、破坏或绕权类表达")
        if any(term.lower() in lower for term in UNAUTHORIZED_TERMS):
            return IntentResult("unauthorized_query", False, 0.9, "命中涉密、隐私或越权调取类表达")
        # Keep the offline fallback conservative: match clear social phrases,
        # but send short topic terms such as "事务锁" through document search.
        casual_text = re.sub(r"[!?？！。，,.、～~\s]+$", "", normalized).lower()
        if casual_text in {
            "你好", "您好", "hello", "hi", "在吗", "早上好", "早安", "中午好",
            "下午好", "晚上好", "晚安", "谢谢", "谢谢你", "多谢", "辛苦了",
            "再见", "拜拜", "回头见", "不客气", "随便聊聊", "陪我聊聊天",
            "讲个笑话", "说个笑话", "你好吗", "你怎么样", "你是谁", "你叫什么",
            "你叫什么名字", "你能做什么", "你会做什么", "你有什么功能", "你在干嘛",
            "今天天气怎么样", "天气怎么样", "现在几点", "今天几号", "今天星期几",
        }:
            return IntentResult("chit_chat", True, 0.74, "普通闲聊或能力咨询")
        if any(term.lower() in lower for term in ASSIGNMENT_TERMS):
            return IntentResult("assignment_query", True, 0.88, "包含作业、练习、答案或解析类诉求")
        if any(term.lower() in lower for term in EXAM_TERMS):
            return IntentResult("exam_preparation", True, 0.87, "包含考试、复习、题型或考点类诉求")
        if any(term.lower() in lower for term in POLICY_TERMS):
            return IntentResult("campus_policy", True, 0.85, "包含资料规则、学分或课程大纲类诉求")
        if any(term.lower() in lower for term in LEARNING_TERMS):
            return IntentResult("course_learning", True, 0.86, "包含文档、章节、概念、定理或学习目标类诉求")
        return IntentResult("document_answer", True, 0.62, "按已上传学科文档问答处理")

    def extract_entities(self, query: str) -> EntityExtractionResult:
        normalized = normalize_text(query)
        entities: Dict[str, List[str]] = {}
        matches = {
            "course_id": self.course_pattern.findall(normalized),
            "grade_level": self.grade_pattern.findall(normalized),
            "knowledge_point": self.knowledge_pattern.findall(normalized),
            "exam_type": self.exam_pattern.findall(normalized),
            "assignment_id": self.assignment_pattern.findall(normalized),
            "score_parameter": self.score_pattern.findall(normalized),
            "learning_node": [node for node in LEARNING_NODES if node in normalized],
        }
        for key, values in matches.items():
            cleaned = unique_keep_order(values)
            if cleaned:
                entities[key] = cleaned
        return EntityExtractionResult(raw_query=query, entities=entities)

    def rewrite_query(self, query: str, intent: IntentResult, entities: EntityExtractionResult) -> str:
        normalized = normalize_text(query)
        expansion: List[str] = [normalized]
        if intent.category in {"assignment_query", "document_answer"}:
            expansion.extend(["学科文档", "资料", "答案", "原文", "解析"])
        elif intent.category == "exam_preparation":
            expansion.extend(["考试", "复习", "考点", "题型"])
        elif intent.category == "campus_policy":
            expansion.extend(["教务", "课程规则", "学分", "培养方案"])
        elif intent.category == "course_learning":
            expansion.extend(["课程", "章节", "知识点", "学习目标"])
        else:
            expansion.extend(["学科文档", "资料", "定义", "原文", "答案"])
        for values in entities.entities.values():
            expansion.extend(values)
        return " ".join(unique_keep_order(expansion))

    def expand_query_variants(
        self,
        query: str,
        intent: IntentResult,
        entities: EntityExtractionResult,
        baseline: str = "",
        max_variants: int = 3,
    ) -> List[str]:
        """Return the original query plus safe semantic alternatives.

        The original query is deliberately retained because a rewrite can lose
        a precise model number, phrase, or user-specific constraint.  This is
        the deterministic Multi-Query fallback used when no LLM agent is
        available.
        """

        normalized = normalize_text(query)
        if not normalized:
            return []
        variants = [normalized, baseline or self.rewrite_query(normalized, intent, entities)]
        lower = normalized.lower()
        for trigger, expansion in QUERY_EXPANSIONS:
            if trigger.lower() in lower:
                variants.append(f"{normalized} {expansion}")
        entity_terms = [value for values in entities.entities.values() for value in values]
        if entity_terms:
            variants.append(" ".join(unique_keep_order(entity_terms + [normalized])))
        return unique_keep_order(variants)[: max(1, int(max_variants))]

    def terms_for_query(self, query: str, entities: EntityExtractionResult) -> List[str]:
        entity_terms: List[str] = []
        for values in entities.entities.values():
            entity_terms.extend(values)
        return unique_keep_order(split_terms(query) + entity_terms)
