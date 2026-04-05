from llm_context.summarizer import summarize, summarize_messages

LONG_TEXT = (
    "The product launch is scheduled for Q3. "
    "The marketing budget has been confirmed at fifty thousand dollars. "
    "The team will focus on LinkedIn and email campaigns targeting HR teams. "
    "ROI tracking will be set up from day one using UTM parameters. "
    "The sales team needs to be briefed two weeks before launch. "
    "Customer success will handle onboarding for the first cohort."
)

def test_summarize_returns_string():
    result = summarize(LONG_TEXT, sentence_count=2)
    assert isinstance(result, str)
    assert len(result) > 0

def test_summarize_respects_count():
    result = summarize(LONG_TEXT, sentence_count=2)
    # a 2-sentence summary should be shorter than original
    assert len(result) < len(LONG_TEXT)

def test_summarize_empty():
    assert summarize("") == ""

def test_summarize_messages():
    msgs = [
        {"role": "user",      "content": "Budget is $50k for Q3 launch."},
        {"role": "assistant", "content": "Noted. LinkedIn and email are the channels."},
    ]
    result = summarize_messages(msgs, sentence_count=2)
    assert isinstance(result, str)