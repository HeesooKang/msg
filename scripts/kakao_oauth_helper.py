#!/usr/bin/env python3
import argparse
import json
import sys
from urllib.parse import urlencode

import requests


AUTHORIZE_URL = "https://kauth.kakao.com/oauth/authorize"
TOKEN_URL = "https://kauth.kakao.com/oauth/token"


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
