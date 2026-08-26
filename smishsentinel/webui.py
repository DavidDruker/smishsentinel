"""A real, user-facing vertical slice around the synthetic inbox.

    inbox message -> background investigation -> quiet/surfaced status ->
    evidence card with clickable sources -> a safe-action call to action

The synthetic inbox and the offline pipeline were already real end to end
(see inbox.py); what was missing was anything a person could actually look
at. This is that: a small local web UI, judges run it themselves, that reads
and writes the exact same ``CaseStore`` ``run_inbox_cycle()`` already uses --
so what it shows is real persisted state, not a mock of one.

Deliberately a separate process from ``app.py``, not more routes bolted onto
it: ``app.py`` is the deployed AgentCore entrypoint, and its HTTP contract is
narrow and CoreBreak-hardened on purpose (see its own docstring) -- adding
unrelated UI routes there would widen exactly the boundary that file exists
to keep tight. This module has no AWS dependency at all: it only reads/writes
the local CaseStore and calls run_inbox_cycle(), which is real Bedrock calls
when actually run, same as everywhere else in this codebase.

Deliberately stdlib-only (``http.server``, no Flask/FastAPI/uvicorn, no
frontend build step) so there is nothing new to install beyond
requirements.txt to see it run.

Run:
    python -m smishsentinel.webui
    # then open http://127.0.0.1:8090/
"""

from __future__ import annotations

import html
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .inbox import run_inbox_cycle
from .schemas import RiskLevel
from .store import CaseRecord, CaseStore, DynamoDBCaseStore, get_case_store

_HOST = "127.0.0.1"
_DEFAULT_PORT = int(os.environ.get("SMISH_WEBUI_PORT", "8090"))

_STATUS_LABEL = {
    "received": "Received",
    "investigating": "Investigating…",
    "complete": "Complete",
    "failed": "Failed",
}

# channel value -> (css class, human label). NONE is the quiet, common case
# by design (see agent.py's triage prompt) -- it should read as unremarkable,
# not as an error state.
_CHANNEL_META = {
    "none": ("quiet", "Quiet — no investigation warranted"),
    "standard": ("standard", "Standard"),
    "urgent": ("urgent", "Urgent"),
}

_RISK_CSS = {
    RiskLevel.HIGH.value: "risk-high",
    RiskLevel.ELEVATED.value: "risk-elevated",
    RiskLevel.UNCLEAR.value: "risk-unclear",
    RiskLevel.QUIET.value: "risk-quiet",
}


# --------------------------------------------------------------------------
# Pure data-shaping helpers -- kept free of any HTTP/HTML concern so they're
# directly unit-testable (see tests/test_webui.py).
# --------------------------------------------------------------------------


def case_summary(record: CaseRecord) -> dict:
    """The list-row shape: just enough to decide whether a case is worth a
    second look, without pulling in the full evidence card."""
    channel = record.notification.channel.value if record.notification else None
    headline = record.card.get("headline") if record.card else None
    warranted = bool(record.triage and record.triage.get("warrants_investigation"))
    return {
        "case_id": record.case_id,
        "received_at": record.received_at,
        "status": record.status.value,
        "message_preview": record.message_text[:120],
        "channel": channel,
        "headline": headline,
        "investigated": warranted,
        "error": record.error,
    }


def recent_case_summaries(store: CaseStore | DynamoDBCaseStore, limit: int = 50) -> list[dict]:
    return [case_summary(r) for r in store.list_recent(limit=limit)]


# --------------------------------------------------------------------------
# HTML rendering -- plain string templates, no templating engine. Every
# value that could contain message-derived text is escaped explicitly;
# nothing here trusts message content to be safe to embed as-is, the same
# stance the rest of this codebase takes toward untrusted input.
# --------------------------------------------------------------------------


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
{_CSS}
</style>
</head>
<body>
{body}
</body>
</html>"""


_CSS = """
:root {
  color-scheme: light dark;
  --bg: #0f1216; --panel: #171b21; --border: #2a2f38; --text: #e6e9ee;
  --muted: #8b93a1; --accent: #4f8cff;
  --quiet: #5b6472; --standard: #4f8cff; --urgent: #e5484d;
  --high: #e5484d; --elevated: #e08a3c; --unclear: #c9a53a; --lowrisk: #5b6472;
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--text); margin: 0;
  font: 15px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif;
}
header {
  padding: 20px 28px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
}
header h1 { font-size: 18px; margin: 0; }
header p { margin: 4px 0 0; color: var(--muted); font-size: 13px; }
main { padding: 24px 28px; max-width: 880px; margin: 0 auto; }
button {
  background: var(--accent); color: white; border: none; border-radius: 6px;
  padding: 10px 16px; font-size: 14px; cursor: pointer;
}
button:disabled { opacity: 0.6; cursor: default; }
a { color: var(--accent); }
.case-row {
  display: block; padding: 14px 16px; margin-bottom: 8px; border-radius: 8px;
  border: 1px solid var(--border); text-decoration: none; color: var(--text);
}
.case-row:hover { border-color: var(--accent); }
.case-row.quiet { opacity: 0.55; }
.case-row-top { display: flex; align-items: center; gap: 10px; justify-content: space-between; }
.badge {
  display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 12px;
  font-weight: 600; text-transform: uppercase; letter-spacing: 0.02em;
}
.badge.quiet { background: color-mix(in srgb, var(--quiet) 30%, transparent); color: var(--muted); }
.badge.standard { background: color-mix(in srgb, var(--standard) 25%, transparent); color: var(--standard); }
.badge.urgent { background: color-mix(in srgb, var(--urgent) 25%, transparent); color: var(--urgent); }
.badge.status { background: var(--panel); color: var(--muted); border: 1px solid var(--border); }
.msg-preview { color: var(--muted); font-size: 13px; margin-top: 6px; }
.headline { margin-top: 6px; font-size: 14px; }
.empty { color: var(--muted); padding: 40px 0; text-align: center; }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 20px 22px; margin-bottom: 16px; }
.card h2 { margin-top: 0; font-size: 16px; }
.risk-pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.risk-high { background: color-mix(in srgb, var(--high) 25%, transparent); color: var(--high); }
.risk-elevated { background: color-mix(in srgb, var(--elevated) 25%, transparent); color: var(--elevated); }
.risk-unclear { background: color-mix(in srgb, var(--unclear) 25%, transparent); color: var(--unclear); }
.risk-quiet { background: color-mix(in srgb, var(--lowrisk) 25%, transparent); color: var(--muted); }
.section-label { font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); margin: 18px 0 8px; }
ul.evidence-list { list-style: none; padding: 0; margin: 0; }
ul.evidence-list li { padding: 10px 0; border-top: 1px solid var(--border); }
ul.evidence-list li:first-child { border-top: none; }
.quote { font-style: italic; color: var(--text); }
.source-url { font-size: 12px; word-break: break-all; }
.first-party { color: #3ecf8e; }
.not-first-party { color: var(--urgent); }
.action-box {
  background: color-mix(in srgb, var(--accent) 12%, var(--panel)); border: 1px solid var(--accent);
  border-radius: 8px; padding: 16px 18px; font-weight: 600; margin-top: 8px;
}
.back-link { display: inline-block; margin-bottom: 16px; color: var(--muted); text-decoration: none; }
.back-link:hover { color: var(--text); }
.mono { font-family: ui-monospace, Consolas, monospace; font-size: 13px; }
"""


def _channel_badge(case: dict) -> tuple[str, str]:
    """(css class, label) for the channel badge -- a failed case reads as
    "investigation failed", not the misleading "not investigated" a crash
    would otherwise share with a message triage genuinely stayed quiet on."""
    if case["status"] == "failed":
        return "urgent", "Investigation failed"
    if case["investigated"]:
        return _CHANNEL_META.get(case["channel"], ("quiet", "—"))
    return "quiet", "Not investigated"


def _render_case_row(case: dict) -> str:
    channel = case["channel"]
    css_class, badge_label = _channel_badge(case)
    row_muted_class = "quiet" if channel in (None, "none") and case["status"] != "failed" else ""
    status_label = _STATUS_LABEL.get(case["status"], case["status"])
    headline_html = (
        f'<div class="headline">{html.escape(case["headline"])}</div>'
        if case["headline"] else ""
    )
    return f"""
<a class="case-row {row_muted_class}" href="/case/{html.escape(case['case_id'])}">
  <div class="case-row-top">
    <span>
      <span class="badge {css_class}">{html.escape(badge_label)}</span>
      <span class="badge status">{html.escape(status_label)}</span>
    </span>
    <span class="mono" style="color: var(--muted); font-size: 12px;">{html.escape(case['case_id'])}</span>
  </div>
  <div class="msg-preview">&ldquo;{html.escape(case['message_preview'])}&rdquo;</div>
  {headline_html}
</a>"""


def render_inbox_page(cases: list[dict]) -> str:
    rows = "".join(_render_case_row(c) for c in cases) or '<div class="empty">No cases yet. Run the demo cycle to populate the inbox.</div>'
    body = f"""
<header>
  <div>
    <h1>SmishSentinel — Inbox</h1>
    <p>Most messages stay quiet on purpose. Only consequential cases surface.</p>
  </div>
  <button id="run-btn" onclick="runCycle()">Run demo inbox cycle</button>
</header>
<main>
  <div id="cases">{rows}</div>
</main>
<script>
async function runCycle() {{
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.textContent = 'Investigating…';
  await fetch('/run-cycle', {{method: 'POST'}});
  poll();
}}
async function poll() {{
  const res = await fetch('/api/cases');
  const cases = await res.json();
  document.getElementById('cases').innerHTML = cases.length ? cases.map(renderRow).join('') :
    '<div class="empty">No cases yet. Run the demo cycle to populate the inbox.</div>';
  const stillWorking = cases.some(c => c.status === 'received' || c.status === 'investigating');
  const btn = document.getElementById('run-btn');
  if (!stillWorking) {{ btn.disabled = false; btn.textContent = 'Run demo inbox cycle'; }}
  if (stillWorking) setTimeout(poll, 1200);
}}
const CHANNEL_META = {{
  none: ['quiet', 'Quiet — no investigation warranted'],
  standard: ['standard', 'Standard'],
  urgent: ['urgent', 'Urgent'],
}};
const STATUS_LABEL = {{received: 'Received', investigating: 'Investigating…', complete: 'Complete', failed: 'Failed'}};
function esc(s) {{
  const d = document.createElement('div'); d.textContent = s ?? ''; return d.innerHTML;
}}
function channelBadge(c) {{
  if (c.status === 'failed') return ['urgent', 'Investigation failed'];
  if (c.investigated) return CHANNEL_META[c.channel] || ['quiet', '—'];
  return ['quiet', 'Not investigated'];
}}
function renderRow(c) {{
  const [cls, badgeLabel] = channelBadge(c);
  const muted = ((!c.channel || c.channel === 'none') && c.status !== 'failed') ? 'quiet' : '';
  const headline = c.headline ? `<div class="headline">${{esc(c.headline)}}</div>` : '';
  return `<a class="case-row ${{muted}}" href="/case/${{esc(c.case_id)}}">
    <div class="case-row-top">
      <span>
        <span class="badge ${{cls}}">${{esc(badgeLabel)}}</span>
        <span class="badge status">${{esc(STATUS_LABEL[c.status] || c.status)}}</span>
      </span>
      <span class="mono" style="color: var(--muted); font-size: 12px;">${{esc(c.case_id)}}</span>
    </div>
    <div class="msg-preview">&ldquo;${{esc(c.message_preview)}}&rdquo;</div>
    ${{headline}}
  </a>`;
}}
poll();
</script>"""
    return _page("SmishSentinel — Inbox", body)


def _render_evidence_item(item: dict) -> str:
    party = (
        '<span class="first-party">first-party</span>' if item.get("is_first_party")
        else '<span class="not-first-party">not verified first-party</span>'
    )
    return f"""<li>
  <div class="quote">&ldquo;{html.escape(item.get('quoted_text', ''))}&rdquo;</div>
  <div class="source-url">
    <a href="{html.escape(item.get('source_url', ''))}" target="_blank" rel="noopener noreferrer">{html.escape(item.get('source_url', ''))}</a>
    &nbsp;&middot;&nbsp;{party}&nbsp;&middot;&nbsp;{html.escape(item.get('source_controller', 'unverified'))}
  </div>
</li>"""


def _render_list(items: list[str]) -> str:
    if not items:
        return '<p class="msg-preview">None.</p>'
    return "<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>"


def render_case_page(record: CaseRecord) -> str:
    status_label = _STATUS_LABEL.get(record.status.value, record.status.value)
    still_working = record.status.value in ("received", "investigating")
    refresh_tag = '<meta http-equiv="refresh" content="2">' if still_working else ""

    header = f"""
<header>
  <div>
    <h1>Case {html.escape(record.case_id)}</h1>
    <p>{html.escape(status_label)} &middot; received {html.escape(record.received_at)}</p>
  </div>
</header>"""

    message_card = f"""
<div class="card">
  <div class="section-label">Original message</div>
  <div class="quote">&ldquo;{html.escape(record.message_text)}&rdquo;</div>
</div>"""

    if record.status.value == "failed":
        body_extra = f"""<div class="card">
  <div class="section-label">Investigation failed</div>
  <p>{html.escape(record.error or 'Unknown error.')}</p>
  <p class="msg-preview">A failure still produces a notification — see notify.py's design note on why silence on failure would be worse than a wrong verdict.</p>
</div>"""
    elif not record.card:
        body_extra = """<div class="card">
  <div class="section-label">Triage</div>
  <p class="msg-preview">This message did not warrant investigation and was left quiet, by design.</p>
</div>"""
    else:
        card = record.card
        risk = card.get("risk_level", "quiet")
        risk_css = _RISK_CSS.get(risk, "risk-quiet")
        evidence_html = (
            "".join(_render_evidence_item(e) for e in card.get("evidence", []))
            or '<p class="msg-preview">No evidence survived verification.</p>'
        )
        body_extra = f"""
<div class="card">
  <div class="section-label">Verdict</div>
  <h2>{html.escape(card.get('headline', ''))}</h2>
  <span class="risk-pill {risk_css}">{html.escape(risk)} risk</span>
  <span class="risk-pill" style="background:none;border:1px solid var(--border);color:var(--muted);">{html.escape(card.get('verdict', ''))}</span>

  <div class="section-label">Observed behaviour</div>
  {_render_list(card.get('observed_behaviour', []))}

  <div class="section-label">Verified facts</div>
  {_render_list(card.get('verified_facts', []))}

  <div class="section-label">Inferences (not verified)</div>
  {_render_list(card.get('inferences', []))}

  <div class="section-label">Unresolved</div>
  {_render_list(card.get('unresolved', []))}

  <div class="section-label">Evidence — clickable sources</div>
  <ul class="evidence-list">{evidence_html}</ul>

  <div class="section-label">What to do</div>
  <div class="action-box">{html.escape(card.get('safe_next_action', ''))}</div>
</div>"""

    body = f"""{refresh_tag}
{header}
<main>
  <a class="back-link" href="/">&larr; Back to inbox</a>
  {message_card}
  {body_extra}
</main>"""
    return _page(f"Case {record.case_id}", body)


# --------------------------------------------------------------------------
# HTTP server -- stdlib only.
# --------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"  # type: ignore[assignment]

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        pass  # keep demo output focused on [notify]/[fetch] pipeline logs

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, html_body: str) -> None:
        self._send(status, "text/html; charset=utf-8", html_body.encode("utf-8"))

    def _send_json(self, status: int, payload) -> None:
        self._send(status, "application/json", json.dumps(payload).encode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        path = urlparse(self.path).path
        store: CaseStore | DynamoDBCaseStore = self.server.store  # type: ignore[attr-defined]

        if path == "/":
            self._send_html(200, render_inbox_page(recent_case_summaries(store)))
            return

        if path == "/api/cases":
            self._send_json(200, recent_case_summaries(store))
            return

        if path.startswith("/case/"):
            case_id = path.removeprefix("/case/")
            record = store.get(case_id)
            if record is None:
                self._send_html(404, _page("Not found", "<main><p>No such case.</p></main>"))
                return
            self._send_html(200, render_case_page(record))
            return

        self._send_html(404, _page("Not found", "<main><p>Not found.</p></main>"))

    def do_POST(self) -> None:  # noqa: N802 - stdlib method name
        if urlparse(self.path).path == "/run-cycle":
            store: CaseStore | DynamoDBCaseStore = self.server.store  # type: ignore[attr-defined]
            # Fire-and-forget in a background thread: run_inbox_cycle persists
            # each case at every stage transition, so the polling UI sees
            # investigations happen live rather than only after they finish.
            # contextvars.ContextVar-based case isolation (tools/evidence.py)
            # is exactly what makes this safe to run off the request thread.
            threading.Thread(target=run_inbox_cycle, kwargs={"store": store}, daemon=True).start()
            self._send_json(202, {"started": True})
            return
        self._send_json(404, {"error": "not found"})


class _Server(ThreadingHTTPServer):
    store: CaseStore | DynamoDBCaseStore


def make_app(
    store: CaseStore | DynamoDBCaseStore | None = None, port: int = _DEFAULT_PORT
) -> _Server:
    server = _Server((_HOST, port), _Handler)
    server.store = store or get_case_store()
    return server


def main() -> None:
    server = make_app()
    port = server.server_address[1]
    print(f"SmishSentinel demo UI: http://{_HOST}:{port}/")
    print("Click \"Run demo inbox cycle\" to investigate the synthetic inbox and watch cases arrive.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
