#!/usr/bin/env python3
import argparse
import json
import os
import secrets
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests


AUTHORIZE_URL = "https://kauth.kakao.com/oauth/authorize"
TOKEN_URL = "https://kauth.kakao.com/oauth/token"


def _strip_env_quotes(value: str) -> str:
    value = str(value or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(path: str) -> dict:
    env_path = Path(path)
    values = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = _strip_env_quotes(value)
    return values


def _env_arg(value: str, env: dict, key: str) -> str:
    return str(value or env.get(key) or os.getenv(key) or "").strip()


def _format_env_line(key: str, value: str) -> str:
    safe = str(value or "").replace("\\", "\\\\").replace('"', '\\"')
    return f'{key}="{safe}"'


def update_env_value(path: str, key: str, value: str) -> None:
    env_path = Path(path)
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    updated = False
    output = []
    for line in lines:
        stripped = line.strip()
        prefix = "export " if stripped.startswith("export ") else ""
        compare = stripped[len("export ") :].strip() if prefix else stripped
        if compare.startswith(f"{key}="):
            output.append(f"{prefix}{_format_env_line(key, value)}")
            updated = True
        else:
            output.append(line)
    if not updated:
        output.append(_format_env_line(key, value))
    env_path.write_text("\n".join(output) + "\n", encoding="utf-8")


def build_auth_url(rest_api_key: str, redirect_uri: str, state: str, scope: str) -> str:
    params = {
        "client_id": rest_api_key,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scope,
    }
    if state:
        params["state"] = state
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


def exchange_code(rest_api_key: str, client_secret: str, redirect_uri: str, code: str) -> dict:
    payload = {
        "grant_type": "authorization_code",
        "client_id": rest_api_key,
        "redirect_uri": redirect_uri,
        "code": code,
    }
    if client_secret:
        payload["client_secret"] = client_secret

    response = requests.post(TOKEN_URL, data=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def refresh_token(rest_api_key: str, client_secret: str, refresh_token_value: str) -> dict:
    payload = {
        "grant_type": "refresh_token",
        "client_id": rest_api_key,
        "refresh_token": refresh_token_value,
    }
    if client_secret:
        payload["client_secret"] = client_secret

    response = requests.post(TOKEN_URL, data=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def _callback_handler(expected_state: str, expected_path: str):
    class KakaoOAuthCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib callback name
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            self.server.auth_code = ""
            self.server.auth_error = ""

            if parsed.path != expected_path:
                self.server.auth_error = f"unexpected callback path: {parsed.path}"
                self._write_response(404, "잘못된 콜백 경로입니다. 터미널로 돌아가 주세요.")
                return

            state = query.get("state", [""])[0]
            if expected_state and state != expected_state:
                self.server.auth_error = "state mismatch"
                self._write_response(400, "인증 state가 일치하지 않습니다. 터미널로 돌아가 주세요.")
                return

            error = query.get("error", [""])[0]
            if error:
                description = query.get("error_description", [""])[0]
                self.server.auth_error = f"{error}: {description}".strip(": ")
                self._write_response(400, "카카오 인증이 실패했습니다. 터미널로 돌아가 주세요.")
                return

            code = query.get("code", [""])[0]
            if not code:
                self.server.auth_error = "authorization code missing"
                self._write_response(400, "인가 코드가 없습니다. 터미널로 돌아가 주세요.")
                return

            self.server.auth_code = code
            self._write_response(200, "카카오 인증이 완료되었습니다. 이 창은 닫으셔도 됩니다.")

        def _write_response(self, status: int, message: str) -> None:
            body = (
                "<!doctype html><meta charset='utf-8'>"
                f"<title>Kakao OAuth</title><p>{message}</p>"
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002,N802 - stdlib callback signature
            return

    return KakaoOAuthCallbackHandler


def _wait_for_authorization_code(redirect_uri: str, auth_url: str, state: str, timeout_seconds: int, open_browser: bool) -> str:
    parsed = urlparse(redirect_uri)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1"} or not parsed.port:
        raise ValueError("setup 명령은 http://localhost:PORT/... 형태의 KAKAO_REDIRECT_URI에서만 자동 수신할 수 있습니다.")

    server = HTTPServer((parsed.hostname, int(parsed.port)), _callback_handler(state, parsed.path or "/"))
    server.timeout = 1
    server.auth_code = ""
    server.auth_error = ""

    print("카카오 인증 URL을 엽니다. 브라우저에서 동의하면 이 터미널이 자동으로 이어서 처리합니다.")
    print(auth_url)
    if open_browser:
        webbrowser.open(auth_url)

    deadline = time.time() + max(10, int(timeout_seconds))
    try:
        while time.time() < deadline:
            server.handle_request()
            if server.auth_code:
                return str(server.auth_code)
            if server.auth_error:
                raise RuntimeError(str(server.auth_error))
    finally:
        server.server_close()

    raise TimeoutError("카카오 인증 대기 시간이 초과되었습니다.")


def setup_refresh_token(
    *,
    env_file: str,
    rest_api_key: str,
    client_secret: str,
    redirect_uri: str,
    scope: str,
    timeout_seconds: int,
    open_browser: bool,
) -> dict:
    state = secrets.token_urlsafe(16)
    auth_url = build_auth_url(rest_api_key, redirect_uri, state, scope)
    code = _wait_for_authorization_code(
        redirect_uri,
        auth_url,
        state,
        timeout_seconds=timeout_seconds,
        open_browser=open_browser,
    )
    body = exchange_code(rest_api_key, client_secret, redirect_uri, code)
    refresh = str(body.get("refresh_token", "") or "").strip()
    if not refresh:
        raise RuntimeError("카카오 응답에 refresh_token이 없습니다.")
    update_env_value(env_file, "KAKAO_REFRESH_TOKEN", refresh)
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description="카카오 OAuth 보조 도구")
    subparsers = parser.add_subparsers(dest="command", required=True)

    auth_parser = subparsers.add_parser("auth-url", help="인가 코드 발급 URL 생성")
    auth_parser.add_argument("--rest-api-key", required=True)
    auth_parser.add_argument("--redirect-uri", required=True)
    auth_parser.add_argument("--state", default="")
    auth_parser.add_argument("--scope", default="talk_message")

    exchange_parser = subparsers.add_parser("exchange-code", help="인가 코드를 토큰으로 교환")
    exchange_parser.add_argument("--rest-api-key", required=True)
    exchange_parser.add_argument("--client-secret", default="")
    exchange_parser.add_argument("--redirect-uri", required=True)
    exchange_parser.add_argument("--code", required=True)

    refresh_parser = subparsers.add_parser("refresh-token", help="refresh_token으로 access_token 갱신")
    refresh_parser.add_argument("--rest-api-key", required=True)
    refresh_parser.add_argument("--client-secret", default="")
    refresh_parser.add_argument("--refresh-token", required=True)

    setup_parser = subparsers.add_parser("setup", help="브라우저 인증 후 .env의 KAKAO_REFRESH_TOKEN 자동 갱신")
    setup_parser.add_argument("--env-file", default=".env")
    setup_parser.add_argument("--rest-api-key", default="")
    setup_parser.add_argument("--client-secret", default="")
    setup_parser.add_argument("--redirect-uri", default="")
    setup_parser.add_argument("--scope", default="talk_message")
    setup_parser.add_argument("--timeout-seconds", type=int, default=180)
    setup_parser.add_argument("--no-browser", action="store_true")
    setup_parser.add_argument("--print-json", action="store_true")

    args = parser.parse_args()

    try:
        if args.command == "auth-url":
            print(build_auth_url(args.rest_api_key, args.redirect_uri, args.state, args.scope))
            return 0

        if args.command == "exchange-code":
            print(
                json.dumps(
                    exchange_code(args.rest_api_key, args.client_secret, args.redirect_uri, args.code),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.command == "refresh-token":
            print(
                json.dumps(
                    refresh_token(args.rest_api_key, args.client_secret, args.refresh_token),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.command == "setup":
            env = load_env_file(args.env_file)
            rest_api_key = _env_arg(args.rest_api_key, env, "KAKAO_REST_API_KEY")
            client_secret = _env_arg(args.client_secret, env, "KAKAO_CLIENT_SECRET")
            redirect_uri = _env_arg(args.redirect_uri, env, "KAKAO_REDIRECT_URI")
            missing = [
                name
                for name, value in {
                    "KAKAO_REST_API_KEY": rest_api_key,
                    "KAKAO_REDIRECT_URI": redirect_uri,
                }.items()
                if not value
            ]
            if missing:
                print(f"필수 값이 없습니다: {', '.join(missing)}", file=sys.stderr)
                return 1
            body = setup_refresh_token(
                env_file=args.env_file,
                rest_api_key=rest_api_key,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=args.scope,
                timeout_seconds=args.timeout_seconds,
                open_browser=not args.no_browser,
            )
            if args.print_json:
                print(json.dumps(body, ensure_ascii=False, indent=2))
            else:
                expires = body.get("refresh_token_expires_in", "")
                suffix = f" refresh_token_expires_in={expires}" if expires else ""
                print(f"{args.env_file}의 KAKAO_REFRESH_TOKEN을 갱신했습니다.{suffix}")
            return 0
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else str(exc)
        print(body, file=sys.stderr)
        return 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
