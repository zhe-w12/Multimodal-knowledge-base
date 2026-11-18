import pytest

from manufacturing_rag.adapters.llm import RuleBasedLLMRouter
from manufacturing_rag.config import RAGConfig
from manufacturing_rag.pipeline import ManufacturingRAGPipeline, contextualize_question
from manufacturing_rag.schemas import DocumentChunk, HallucinationCheck, RetrievalResult


SCHOLARSHIP_HISTORY = (
    "用户：奖学金的奖金\n"
    "助手：你想了解哪类奖学金？\n"
    "用户：校级优秀学生奖学金\n"
    "助手：请问是2025学年还是2026学年、哪个学院，以及评选条件？"
)


def test_scholarship_supplement_keeps_amount_goal_without_assistant_conditions():
    effective = contextualize_question("学业优等生奖学金", SCHOLARSHIP_HISTORY)

    assert "奖金" in effective
    assert "校级优秀学生奖学金" in effective
    assert "学业优等生奖学金" in effective
    assert "2025" not in effective
    assert "2026" not in effective
    assert "学院" not in effective
    assert "评选条件" not in effective


@pytest.mark.parametrize("question", ["缓考怎么申请？", "极限的定义是什么？", "图书馆在哪里？", "图书馆几点关门？"])
def test_complete_new_question_does_not_inherit_scholarship_topic(question):
    assert contextualize_question(question, SCHOLARSHIP_HISTORY) == question


def test_supplement_uses_latest_user_goal_after_a_topic_change():
    history = "用户：极限的定义是什么？\n助手：极限是函数值的变化趋势。\n" + SCHOLARSHIP_HISTORY

    effective = contextualize_question("学业优等生奖学金", history)

    assert "奖金" in effective
    assert "极限" not in effective


def test_referential_followup_retains_answer_as_reference_but_not_assistant_question():
    effective = contextualize_question(
        "第二个的奖金是多少？",
        "用户：校级奖学金有哪些？\n助手：第一项是优秀学生奖学金，第二项是学业优等生奖学金。",
    )

    assert "学业优等生奖学金" in effective
    assert "不作为用户条件" in effective
    assert "2026" not in contextualize_question("这个呢？", "用户：奖学金的奖金\n助手：你要问2026学年吗？")


def test_unlabelled_history_remains_usable():
    effective = contextualize_question("学业优等生奖学金", "奖学金的奖金；校级优秀学生奖学金")

    assert "奖金" in effective
    assert "学业优等生奖学金" in effective


@pytest.mark.parametrize("question", ["能详细解释一下吗？", "再说明一下", "怎么改？", "为什么会这样？"])
def test_elliptical_followup_keeps_previous_goal(question):
    effective = contextualize_question(question, SCHOLARSHIP_HISTORY)
    assert "奖金" in effective
    assert question in effective
    assert "2025" not in effective


def build_recording_pipeline(*, sufficient=False, resolved_question=None):
    config = RAGConfig(
        llm_provider="rule", embedding_provider="hash", rerank_provider="rule",
        vector_store_provider="memory", query_classifier_provider="rule", query_agent_provider="rule",
        redis_enabled=False, mysql_enabled=False, embedding_dimensions=64, debug_log_enabled=False,
        rerank_threshold=0.0,
    )

    class RecordingRouter(RuleBasedLLMRouter):
        def __init__(self):
            super().__init__(config)
            self.calls = {}

        def resolve_follow_up_question(self, question, conversation_context):
            self.calls["resolve"] = (question, conversation_context)
            return resolved_question

        def assess_context_sufficiency(self, question, contexts):
            self.calls["sufficiency"] = question
            return sufficient, "检索资料未列出奖金金额"

        def generate_clarifying_question(self, question, contexts, reason):
            self.calls["clarify"] = question
            return "请补充包含奖金金额的资料。"

        def generate_clarifying_question_stream(self, question, contexts, reason, on_token):
            self.calls["clarify_stream"] = question
            answer = "请补充包含奖金金额的资料。"
            on_token(answer)
            return answer

        def generate_answer(self, question, contexts):
            self.calls["answer"] = question
            return "奖金为1000元。"

        def generate_answer_stream(self, question, contexts, on_token):
            self.calls["answer_stream"] = question
            answer = "奖金为1000元。"
            on_token(answer)
            return answer

        def hallucination_check(self, answer, source_texts):
            return HallucinationCheck(True, [], 1.0)

    class Retriever:
        def search(self, question, entities, limit):
            return [RetrievalResult(DocumentChunk(
                id="scholarship", source="学生手册.pdf", text="学业优等生奖学金奖金为1000元。",
            ), 0.99)]

    router = RecordingRouter()
    pipeline = ManufacturingRAGPipeline([], config=config, retriever=Retriever(), llm_router=router)
    return pipeline, router


@pytest.mark.parametrize("stream", [False, True])
def test_sufficiency_and_clarification_receive_resolved_amount_question(stream):
    resolved = "校级优秀学生奖学金中的学业优等生奖学金，奖金是多少？"
    pipeline, router = build_recording_pipeline(resolved_question=resolved)
    tokens = []

    response = pipeline.answer(
        "学业优等生奖学金", conversation_context=SCHOLARSHIP_HISTORY,
        on_token=tokens.append if stream else None,
    )

    assert router.calls["resolve"] == ("学业优等生奖学金", SCHOLARSHIP_HISTORY)
    assert router.calls["sufficiency"] == resolved
    assert router.calls["clarify_stream" if stream else "clarify"] == resolved
    assert response.question == "学业优等生奖学金"
    if stream:
        assert "".join(tokens) == response.answer


@pytest.mark.parametrize("stream", [False, True])
def test_generation_receives_same_resolved_question_as_sufficiency(stream):
    resolved = "学业优等生奖学金的奖金是多少？"
    pipeline, router = build_recording_pipeline(sufficient=True, resolved_question=resolved)

    pipeline.answer(
        "学业优等生奖学金", conversation_context=SCHOLARSHIP_HISTORY,
        on_token=(lambda token: None) if stream else None,
    )

    assert router.calls["sufficiency"] == resolved
    assert router.calls["answer_stream" if stream else "answer"] == resolved


def test_empty_resolver_output_falls_back_to_user_context():
    pipeline, router = build_recording_pipeline(resolved_question="  ")

    pipeline.answer("学业优等生奖学金", conversation_context=SCHOLARSHIP_HISTORY)

    effective = router.calls["clarify"]
    assert effective == router.calls["sufficiency"]
    assert "奖金" in effective
    assert "2025" not in effective


def test_independent_new_question_does_not_invoke_followup_resolver():
    pipeline, router = build_recording_pipeline()

    pipeline.answer("缓考怎么申请？", conversation_context=SCHOLARSHIP_HISTORY)

    assert "resolve" not in router.calls
    assert router.calls["sufficiency"] == "缓考怎么申请？"
