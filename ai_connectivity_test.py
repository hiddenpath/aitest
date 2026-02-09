"""
AI Provider Connectivity & Functionality Test Script
=====================================================
Tests all available AI providers using their API keys from environment variables.
Uses httpx (async) for direct API calls to each provider.

Tests performed per provider:
  1. Connectivity (can we reach the API endpoint?)
  2. Basic Chat Completion (non-streaming)
  3. Streaming Chat Completion
"""

import asyncio
import os
import json
import time
import sys
from dataclasses import dataclass, field

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import httpx

# ─── Test Configuration ──────────────────────────────────────────────────────

TEST_PROMPT = "请用一句话回答：1+1等于几？"
TIMEOUT = 30.0
STREAM_TIMEOUT = 45.0
SOCKS_PROXY = "socks5://192.168.2.13:8889"

# Providers that need SOCKS proxy (blocked in China)
NEEDS_PROXY = {"OpenAI", "Google Gemini", "Cohere", "Groq"}

# ─── Provider Definitions ────────────────────────────────────────────────────

@dataclass
class ProviderConfig:
    name: str
    env_key: str
    base_url: str
    chat_endpoint: str
    model: str
    auth_style: str = "bearer"
    api_key_param: str = ""
    models_endpoint: str = ""
    streaming_supported: bool = True
    request_builder: str = "openai"  # openai | gemini | cohere

PROVIDERS = [
    ProviderConfig(
        name="OpenAI",
        env_key="OPENAI_API_KEY",
        base_url="https://api.openai.com",
        chat_endpoint="/v1/chat/completions",
        model="gpt-4o-mini",
        models_endpoint="/v1/models",
    ),
    ProviderConfig(
        name="DeepSeek",
        env_key="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        chat_endpoint="/v1/chat/completions",
        model="deepseek-chat",
        models_endpoint="/v1/models",
    ),
    ProviderConfig(
        name="Groq",
        env_key="GROQ_API_KEY",
        base_url="https://api.groq.com/openai",
        chat_endpoint="/v1/chat/completions",
        model="llama-3.3-70b-versatile",
        models_endpoint="/v1/models",
    ),
    ProviderConfig(
        name="Mistral",
        env_key="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai",
        chat_endpoint="/v1/chat/completions",
        model="mistral-small-latest",
        models_endpoint="/v1/models",
    ),
    ProviderConfig(
        name="ZhiPu (智谱清言)",
        env_key="ZHIPU_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas",
        chat_endpoint="/v4/chat/completions",
        model="glm-4-flash",
    ),
    ProviderConfig(
        name="MiniMax",
        env_key="MINIMAX_API_KEY",
        base_url="https://api.minimax.chat",
        chat_endpoint="/v1/text/chatcompletion_v2",
        model="MiniMax-Text-01",
    ),
    ProviderConfig(
        name="Google Gemini",
        env_key="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com",
        chat_endpoint="/v1beta/models/{model}:generateContent",
        model="gemini-2.0-flash",
        auth_style="query_param",
        api_key_param="key",
        request_builder="gemini",
        models_endpoint="/v1beta/models",
    ),
    ProviderConfig(
        name="Cohere",
        env_key="COHERE_API_KEY",
        base_url="https://api.cohere.com",
        chat_endpoint="/v2/chat",
        model="command-r",
        request_builder="cohere",
        models_endpoint="/v2/models",
    ),
]

# ─── Test Result Types ───────────────────────────────────────────────────────

@dataclass
class TestResult:
    test_name: str
    success: bool
    latency_ms: float = 0.0
    detail: str = ""
    error: str = ""

@dataclass
class ProviderReport:
    provider_name: str
    api_key_present: bool = False
    used_proxy: bool = False
    results: list = field(default_factory=list)

# ─── Request Builders ────────────────────────────────────────────────────────

def build_openai_request(model: str, prompt: str, stream: bool = False) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.1,
        "stream": stream,
    }

def build_gemini_request(prompt: str) -> dict:
    return {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 100, "temperature": 0.1},
    }

def build_cohere_request(model: str, prompt: str, stream: bool = False) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.1,
        "stream": stream,
    }

# ─── Response Parsers ────────────────────────────────────────────────────────

def parse_openai_response(data: dict) -> tuple[str, str]:
    """Returns (content, usage_info)"""
    content = ""
    choices = data.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "<empty>")
    else:
        content = f"<unexpected: {json.dumps(data, ensure_ascii=False)[:150]}>"
    usage = data.get("usage", {})
    usage_info = ""
    if usage:
        pt = usage.get("prompt_tokens", usage.get("input_tokens", "?"))
        ct = usage.get("completion_tokens", usage.get("output_tokens", "?"))
        usage_info = f"tokens: {pt}->{ct}"
    return content.strip(), usage_info

def parse_gemini_response(data: dict) -> tuple[str, str]:
    candidates = data.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            return parts[0].get("text", "<empty>").strip(), ""
    return f"<unexpected: {json.dumps(data, ensure_ascii=False)[:150]}>", ""

def parse_cohere_response(data: dict) -> tuple[str, str]:
    message = data.get("message", {})
    content = ""
    if message:
        content_list = message.get("content", [])
        if content_list:
            content = content_list[0].get("text", "<empty>")
    usage = data.get("usage", {})
    usage_info = ""
    if usage:
        tin = usage.get("tokens", {}).get("input_tokens", "?")
        tout = usage.get("tokens", {}).get("output_tokens", "?")
        usage_info = f"tokens: {tin}->{tout}"
    if not content:
        content = f"<unexpected: {json.dumps(data, ensure_ascii=False)[:150]}>"
    return content.strip(), usage_info

# ─── Client Factory ──────────────────────────────────────────────────────────

def create_client(use_proxy: bool = False) -> httpx.AsyncClient:
    """Create an httpx AsyncClient, optionally with SOCKS proxy."""
    if use_proxy:
        return httpx.AsyncClient(
            proxy=SOCKS_PROXY,
            follow_redirects=True,
            verify=True,
        )
    else:
        transport = httpx.AsyncHTTPTransport(retries=1)
        return httpx.AsyncClient(
            transport=transport,
            follow_redirects=True,
            verify=True,
        )

# ─── Core Test Functions ─────────────────────────────────────────────────────

async def test_connectivity(client: httpx.AsyncClient, provider: ProviderConfig, api_key: str) -> TestResult:
    """Test basic connectivity to the provider's API endpoint."""
    start = time.perf_counter()
    try:
        # Try models endpoint if available, otherwise just hit base URL
        if provider.models_endpoint:
            url = provider.base_url + provider.models_endpoint
        else:
            # For providers without models endpoint, try a HEAD to chat endpoint
            url = provider.base_url + provider.chat_endpoint
            if "{model}" in url:
                url = url.replace("{model}", provider.model)

        params = {}
        headers = {}
        if provider.auth_style == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        elif provider.auth_style == "query_param":
            params[provider.api_key_param] = api_key

        resp = await client.get(url, headers=headers, params=params, timeout=TIMEOUT)
        latency = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            try:
                data = resp.json()
                # Try to extract model names
                models_list = data.get("data", data.get("models", []))
                if isinstance(models_list, list) and models_list:
                    model_names = [m.get("id", m.get("name", "?")) for m in models_list[:5]]
                    detail = f"HTTP 200 | 可用模型(前5): {', '.join(str(n) for n in model_names)}"
                else:
                    detail = f"HTTP 200 | 端点可达"
            except Exception:
                detail = f"HTTP 200 | 端点可达"
            return TestResult("连通性测试", True, latency, detail)
        elif resp.status_code in (401, 403):
            return TestResult("连通性测试", False, latency,
                              error=f"HTTP {resp.status_code} 认证失败 (API Key可能无效或过期)")
        elif resp.status_code == 404:
            # Some providers don't have a models endpoint, but are still reachable
            return TestResult("连通性测试", True, latency,
                              detail=f"HTTP {resp.status_code} | 端点可达(无模型列表接口)")
        else:
            return TestResult("连通性测试", False, latency,
                              error=f"HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return TestResult("连通性测试", False, latency, error=str(e)[:300])


async def test_chat_completion(client: httpx.AsyncClient, provider: ProviderConfig, api_key: str) -> TestResult:
    """Test basic (non-streaming) chat completion."""
    start = time.perf_counter()
    try:
        url = provider.base_url + provider.chat_endpoint
        if "{model}" in url:
            url = url.replace("{model}", provider.model)

        headers = {"Content-Type": "application/json"}
        params = {}
        if provider.auth_style == "bearer":
            headers["Authorization"] = f"Bearer {api_key}"
        elif provider.auth_style == "query_param":
            params[provider.api_key_param] = api_key

        if provider.request_builder == "gemini":
            body = build_gemini_request(TEST_PROMPT)
        elif provider.request_builder == "cohere":
            body = build_cohere_request(provider.model, TEST_PROMPT, stream=False)
        else:
            body = build_openai_request(provider.model, TEST_PROMPT, stream=False)

        resp = await client.post(url, json=body, headers=headers, params=params, timeout=TIMEOUT)
        latency = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()

            # Check for provider-specific error in body
            if "base_resp" in data:
                status_code = data["base_resp"].get("status_code", 0)
                status_msg = data["base_resp"].get("status_msg", "")
                if status_code != 0:
                    return TestResult("基础对话补全", False, latency,
                                      error=f"API错误 code={status_code}: {status_msg}")

            if provider.request_builder == "gemini":
                content, usage_info = parse_gemini_response(data)
            elif provider.request_builder == "cohere":
                content, usage_info = parse_cohere_response(data)
            else:
                content, usage_info = parse_openai_response(data)

            detail = f"模型: {provider.model} | 回复: {content[:80]}"
            if usage_info:
                detail += f" | {usage_info}"
            return TestResult("基础对话补全", True, latency, detail)
        else:
            error_text = resp.text[:300]
            try:
                err_json = resp.json()
                error_msg = err_json.get("error", {}).get("message", error_text)
                error_text = error_msg[:300]
            except Exception:
                pass
            return TestResult("基础对话补全", False, latency,
                              error=f"HTTP {resp.status_code}: {error_text}")
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return TestResult("基础对话补全", False, latency, error=str(e)[:300])


async def test_streaming(client: httpx.AsyncClient, provider: ProviderConfig, api_key: str) -> TestResult:
    """Test streaming chat completion."""
    if not provider.streaming_supported:
        return TestResult("流式输出", False, 0, error="该提供商不支持流式输出")

    start = time.perf_counter()
    try:
        url = provider.base_url + provider.chat_endpoint
        headers = {"Content-Type": "application/json"}
        params = {}

        if provider.request_builder == "gemini":
            url = provider.base_url + f"/v1beta/models/{provider.model}:streamGenerateContent"
            params[provider.api_key_param] = api_key
            params["alt"] = "sse"
            body = build_gemini_request(TEST_PROMPT)
        elif provider.request_builder == "cohere":
            headers["Authorization"] = f"Bearer {api_key}"
            body = build_cohere_request(provider.model, TEST_PROMPT, stream=True)
        else:
            if "{model}" in url:
                url = url.replace("{model}", provider.model)
            headers["Authorization"] = f"Bearer {api_key}"
            body = build_openai_request(provider.model, TEST_PROMPT, stream=True)

        chunks_received = 0
        collected_text = ""
        first_chunk_latency = None

        async with client.stream("POST", url, json=body, headers=headers, params=params, timeout=STREAM_TIMEOUT) as resp:
            if resp.status_code != 200:
                body_text = ""
                async for chunk in resp.aiter_text():
                    body_text += chunk
                    if len(body_text) > 300:
                        break
                latency = (time.perf_counter() - start) * 1000
                return TestResult("流式输出", False, latency, error=f"HTTP {resp.status_code}: {body_text[:300]}")

            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        chunks_received += 1
                        if first_chunk_latency is None:
                            first_chunk_latency = (time.perf_counter() - start) * 1000

                        if provider.request_builder == "gemini":
                            candidates = data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                if parts:
                                    collected_text += parts[0].get("text", "")
                        elif provider.request_builder == "cohere":
                            evt_type = data.get("type", "")
                            if evt_type == "content-delta":
                                delta_text = data.get("delta", {}).get("message", {}).get("content", {}).get("text", "")
                                collected_text += delta_text
                        else:
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                collected_text += delta.get("content", "")
                    except json.JSONDecodeError:
                        pass

        total_latency = (time.perf_counter() - start) * 1000
        if chunks_received > 0:
            detail = (
                f"收到 {chunks_received} 个chunk | "
                f"首chunk: {first_chunk_latency:.0f}ms | "
                f"总耗时: {total_latency:.0f}ms | "
                f"内容: {collected_text.strip()[:60]}"
            )
            return TestResult("流式输出", True, total_latency, detail)
        else:
            return TestResult("流式输出", False, total_latency, error="未收到任何有效chunk数据")

    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return TestResult("流式输出", False, latency, error=str(e)[:300])

# ─── Print Helpers ───────────────────────────────────────────────────────────

def print_result(result: TestResult):
    status = "[PASS]" if result.success else "[FAIL]"
    print(f"  {status} {result.test_name} | {result.latency_ms:.0f}ms")
    if result.success:
        print(f"         {result.detail}")
    else:
        print(f"         错误: {result.error}")

# ─── Main Test Runner ────────────────────────────────────────────────────────

async def run_provider_tests(provider: ProviderConfig, api_key: str) -> ProviderReport:
    """Run all tests for a single provider."""
    report = ProviderReport(provider_name=provider.name, api_key_present=True)
    use_proxy = provider.name in NEEDS_PROXY

    if use_proxy:
        report.used_proxy = True

    async with create_client(use_proxy=use_proxy) as client:
        # Test 1: Connectivity
        r1 = await test_connectivity(client, provider, api_key)
        report.results.append(r1)
        print_result(r1)

        # If connectivity totally failed, still try other tests
        # Test 2: Chat Completion
        r2 = await test_chat_completion(client, provider, api_key)
        report.results.append(r2)
        print_result(r2)

        # Test 3: Streaming
        r3 = await test_streaming(client, provider, api_key)
        report.results.append(r3)
        print_result(r3)

    return report


async def run_all_tests():
    print("=" * 78)
    print("  AI Provider 连通性 & 功能性测试报告")
    print(f"  测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]} | httpx: {httpx.__version__}")
    print(f"  SOCKS代理: {SOCKS_PROXY}")
    print(f"  需要代理的提供商: {', '.join(NEEDS_PROXY)}")
    print("=" * 78)

    # SDK validation
    print(f"\n{'─' * 78}")
    print(f"  [SDK] ai-lib-python 包验证")
    print(f"{'─' * 78}")
    try:
        from ai_lib_python import AiClient, Message
        from ai_lib_python.resilience import RetryConfig
        msg = Message.user(TEST_PROMPT)
        print(f"  [PASS] ai-lib-python v0.5.0 导入成功 | Message.role={msg.role}")
    except Exception as e:
        print(f"  [FAIL] ai-lib-python 导入失败: {e}")

    reports: list[ProviderReport] = []

    for provider in PROVIDERS:
        api_key = os.environ.get(provider.env_key, "")

        print(f"\n{'─' * 78}")
        proxy_tag = " [PROXY]" if provider.name in NEEDS_PROXY else " [DIRECT]"
        print(f"  [{provider.name}]{proxy_tag} env={provider.env_key} | model={provider.model}")
        print(f"{'─' * 78}")

        if not api_key:
            print(f"  [SKIP] API Key 未设置 ({provider.env_key})")
            reports.append(ProviderReport(provider_name=provider.name))
            continue

        report = await run_provider_tests(provider, api_key)
        reports.append(report)

    # ─── Summary Table ────────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("  综合测试报告摘要")
    print(f"{'=' * 78}")

    header = f"  {'提供商':<20} {'代理':^6} {'连通性':^8} {'对话':^8} {'流式':^8} {'均延迟':>8}"
    print(f"\n{header}")
    print(f"  {'─' * 68}")

    total_pass = 0
    total_fail = 0
    total_skip = 0

    for report in reports:
        if not report.api_key_present:
            print(f"  {report.provider_name:<20} {'':^6} {'跳过':^8} {'跳过':^8} {'跳过':^8} {'N/A':>8}")
            total_skip += 1
            continue

        proxy_flag = "是" if report.used_proxy else "否"
        row = f"  {report.provider_name:<20} {proxy_flag:^6}"
        latencies = []
        for r in report.results:
            if r.success:
                row += f" {'PASS':^8}"
                total_pass += 1
                latencies.append(r.latency_ms)
            else:
                row += f" {'FAIL':^8}"
                total_fail += 1

        avg_lat = f"{sum(latencies) / len(latencies):.0f}ms" if latencies else "N/A"
        row += f" {avg_lat:>8}"
        print(row)

    print(f"\n  总计: {total_pass} 通过 | {total_fail} 失败 | {total_skip} 跳过 (无API Key)")

    # Provider-level summary
    print(f"\n  按提供商汇总:")
    for report in reports:
        if not report.api_key_present:
            continue
        passed = sum(1 for r in report.results if r.success)
        total = len(report.results)
        status = "完全可用" if passed == total else f"部分可用({passed}/{total})" if passed > 0 else "不可用"
        print(f"    {report.provider_name:<20} -> {status}")

    print(f"\n{'=' * 78}")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
