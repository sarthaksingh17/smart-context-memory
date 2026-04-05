from llm_context.retriever import get_relevant_messages, build_retrieval_index, query_index

HISTORY = [
    {"role": "user",      "content": "Hi, I need help with a product launch."},
    {"role": "assistant", "content": "Happy to help! What's the product?"},
    {"role": "user",      "content": "B2B SaaS tool for HR. Budget is $50k."},
    {"role": "assistant", "content": "Great. What channels are you using?"},
    {"role": "user",      "content": "LinkedIn and email campaigns."},
    {"role": "assistant", "content": "Good choices for B2B. Any ROI targets?"},
]

def test_returns_list():
    result = get_relevant_messages("budget", HISTORY, top_k=3)
    assert isinstance(result, list)

def test_top_k_respected():
    result = get_relevant_messages("budget", HISTORY, top_k=2)
    assert len(result) <= 4  # 2 + always_include_last_n=2

def test_empty_history():
    assert get_relevant_messages("anything", [], top_k=3) == []

def test_empty_query_returns_tail():
    result = get_relevant_messages("", HISTORY, top_k=3)
    assert result == HISTORY[-3:]

def test_index_query_consistent():
    index   = build_retrieval_index(HISTORY)
    direct  = get_relevant_messages("marketing budget", HISTORY, top_k=3, always_include_last_n=0)
    indexed = query_index("marketing budget", index, top_k=3)
    assert direct == indexed
