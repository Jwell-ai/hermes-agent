from types import SimpleNamespace

from agent.chat_completion_helpers import build_api_kwargs


def test_anthropic_request_overrides_merge_extra_headers():
    class _Transport:
        def build_kwargs(self, **kwargs):
            return {
                "model": kwargs["model"],
                "messages": kwargs["messages"],
                "extra_headers": {"anthropic-beta": "context-1m"},
            }

    agent = SimpleNamespace(
        api_mode="anthropic_messages",
        tools=[],
        model="claude-sonnet-4-6",
        max_tokens=4096,
        reasoning_config=None,
        request_overrides={
            "extra_headers": {
                "X-App-Secret": "service-secret",
                "X-Internal-User-ID": "42",
            }
        },
        _is_anthropic_oauth=False,
        _anthropic_base_url="http://jwell/internal",
        _prepare_anthropic_messages_for_api=lambda messages: messages,
        _anthropic_preserve_dots=lambda: False,
        _get_transport=lambda: _Transport(),
        context_compressor=SimpleNamespace(context_length=200_000),
    )

    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hello"}])

    assert kwargs["extra_headers"] == {
        "anthropic-beta": "context-1m",
        "X-App-Secret": "service-secret",
        "X-Internal-User-ID": "42",
    }
