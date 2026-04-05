from llm_context.memory import ContextManager

def test_add_and_get():
    cm = ContextManager(max_tokens=4000)
    cm.add_message("user", "Hello!")
    ctx = cm.get_context()
    assert len(ctx) >= 1

def test_clear():
    cm = ContextManager()
    cm.add_message("user", "Test message")
    cm.clear()
    assert cm.messages == []
    assert cm.summary == ""

def test_token_usage_keys():
    cm = ContextManager()
    cm.add_message("user", "Some content here.")
    usage = cm.token_usage()
    assert all(k in usage for k in
               ["messages_tokens", "summary_tokens", "total", "max", "usage_percent"])

def test_compression_triggers():
    # Use a tiny token budget to force compression
    cm = ContextManager(max_tokens=50)
    for i in range(20):
        cm.add_message("user", f"Message number {i} with some extra words to use tokens.")
    # After compression, summary should be populated
    assert cm.summary != "" or len(cm.messages) < 20

def test_get_context_with_query():
    cm = ContextManager(max_tokens=4000)
    cm.add_message("user", "The budget is $50k for the Q3 launch.")
    cm.add_message("assistant", "Noted. Any specific channels in mind?")
    cm.add_message("user", "LinkedIn and email.")
    ctx = cm.get_context(query="What is the budget?", top_k=2)
    assert isinstance(ctx, list)
