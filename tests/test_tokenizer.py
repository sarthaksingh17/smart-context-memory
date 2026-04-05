from llm_context.tokenizer import count_tokens, count_messages_tokens, token_ratio

def test_count_tokens_basic():
    assert count_tokens("Hello world") > 0

def test_count_tokens_empty():
    assert count_tokens("") == 0

def test_count_messages_tokens():
    msgs = [
        {"role": "user",      "content": "What is the price?"},
        {"role": "assistant", "content": "It costs $50."},
    ]
    total = count_messages_tokens(msgs)
    assert total > 10  # must have overhead

def test_token_ratio():
    r = token_ratio("short", "this is a much longer sentence for comparison")
    assert r < 1.0  # short is fewer tokens
