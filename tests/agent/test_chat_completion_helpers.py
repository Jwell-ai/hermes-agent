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

    assert kwargs["extra_headers"]["anthropic-beta"] == "context-1m"
    assert kwargs["extra_headers"]["X-App-Secret"] == "service-secret"
    assert kwargs["extra_headers"]["X-Internal-User-ID"] == "42"
    assert kwargs["extra_headers"]["Idempotency-Key"].startswith("hermes-text:")


def test_provider_profile_request_overrides_include_relay_idempotency_key():
    class _Transport:
        def build_kwargs(self, **kwargs):
            return kwargs

    agent = SimpleNamespace(
        api_mode="chat_completions",
        tools=[],
        model="deepseek-chat",
        provider="deepseek",
        base_url="http://jwell/internal",
        _base_url_lower="http://jwell/internal",
        _base_url_hostname="jwell",
        max_tokens=4096,
        reasoning_config=None,
        request_overrides={
            "extra_headers": {
                "X-App-Secret": "service-secret",
                "X-Internal-User-ID": "42",
            }
        },
        session_id="session-1",
        providers_allowed=[],
        providers_ignored=[],
        providers_order=[],
        provider_sort=None,
        provider_require_parameters=False,
        provider_data_collection=None,
        _is_qwen_portal=lambda: False,
        _is_openrouter_url=lambda: False,
        _get_transport=lambda: _Transport(),
        _resolved_api_call_timeout=lambda: 60,
        _max_tokens_param=lambda: "max_tokens",
        _supports_reasoning_extra_body=lambda: True,
        _prepare_messages_for_non_vision_model=lambda messages: messages,
        _ollama_num_ctx=None,
        openrouter_min_coding_score=None,
    )

    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hello"}])

    assert kwargs["request_overrides"]["extra_headers"]["Idempotency-Key"].startswith("hermes-text:")
