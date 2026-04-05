from .tokenizer import count_tokens, count_messages_tokens
from .summarizer import summarize
from .retriever import get_relevant_messages


class ContextManager:
    """
    Smart context/memory manager for LLM conversations.
    Keeps your prompt within token limits automatically —
    summarizing old messages and retrieving only what's relevant.
    """

    def __init__(self, max_tokens: int = 4000, summary_ratio: float = 0.3):
        self.max_tokens = max_tokens
        self.summary_ratio = summary_ratio
        self.messages = []
        self.summary = ""

    def add_message(self, role: str, content: str):
        """Add a new message to history."""
        self.messages.append({"role": role, "content": content})
        self._maybe_compress()

    def get_context(self, query: str = None, top_k: int = 6) -> list:
        """
        Get the smart context to send to your LLM.
        If query provided, retrieves semantically relevant messages.
        Always prepends the rolling summary if one exists.
        """
        context = []

        if self.summary:
            context.append({
                "role": "system",
                "content": f"Summary of earlier conversation: {self.summary}"
            })

        if query:
            relevant = get_relevant_messages(query, self.messages, top_k=top_k)
            context.extend(relevant)
        else:
            context.extend(self.messages)

        return context

    def _maybe_compress(self):
        """If we exceed max_tokens, summarize the oldest messages."""
        total = count_messages_tokens(self.messages)
        if total <= self.max_tokens:
            return

        cutoff = len(self.messages) // 2
        to_compress = self.messages[:cutoff]
        self.messages = self.messages[cutoff:]

        old_text = " ".join(m["content"] for m in to_compress)
        if self.summary:
            old_text = self.summary + " " + old_text

        sentence_count = max(2, int(len(to_compress) * self.summary_ratio))
        self.summary = summarize(old_text, sentence_count=sentence_count)

    def clear(self):
        """Reset everything."""
        self.messages = []
        self.summary = ""

    def token_usage(self) -> dict:
        """See how many tokens you're currently using."""
        msg_tokens = count_messages_tokens(self.messages)
        summary_tokens = count_tokens(self.summary) if self.summary else 0
        return {
            "messages_tokens": msg_tokens,
            "summary_tokens": summary_tokens,
            "total": msg_tokens + summary_tokens,
            "max": self.max_tokens,
            "usage_percent": round(
                (msg_tokens + summary_tokens) / self.max_tokens * 100, 1
            ) if self.max_tokens > 0 else 0.0
        }