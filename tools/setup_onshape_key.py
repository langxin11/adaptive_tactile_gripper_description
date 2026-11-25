#!/usr/bin/env python3
"""交互式配置 Onshape API 密钥，校验连通性并安全写入本地环境配置。"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import json
import os
import secrets
import shlex
import string
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_API_URL = "https://cad.onshape.com"
DEFAULT_DOC_ID = "ee6a871f9e00fc227503180b"
DEFAULT_OUTPUT_PATH = Path.home() / ".config" / "onshape" / "credentials.env"
DEFAULT_BASHRC_PATH = Path.home() / ".bashrc"


def build_onshape_headers(
    method: str,
    path: str,
    access_key: str,
    secret_key: str,
    query: str = "",
    content_type: str = "application/json",
) -> dict[str, str]:
    """按 Onshape HMAC-SHA256 规范生成带签名的请求头。"""
    alphabet = string.ascii_uppercase + string.digits
    nonce = "".join(secrets.choice(alphabet) for _ in range(25))
    date_str = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    # 签名串末尾必须保留换行符，且整体转为小写后计算 HMAC-SHA256
    hmac_str = (
        f"{method}\n{nonce}\n{date_str}\n{content_type}\n{path}\n{query}\n"
    ).lower().encode("utf-8")
    digest = hmac.new(secret_key.encode("utf-8"), hmac_str, digestmod=hashlib.sha256).digest()
    signature = base64.b64encode(digest).decode("utf-8")

    return {
        "Content-Type": content_type,
        "Date": date_str,
        "On-Nonce": nonce,
        "Accept": "application/json",
        "User-Agent": "adaptive-tactile-gripper-setup",
        "Authorization": f"On {access_key}:HmacSHA256:{signature}",
    }


def verify_credentials(
    api_url: str, doc_id: str, access_key: str, secret_key: str
) -> tuple[bool, str]:
    """请求目标 Onshape 文档接口，验证密钥有效性及访问权限。"""
    base = api_url.rstrip("/")
    path = f"/api/v6/documents/{doc_id}"
    headers = build_onshape_headers("GET", path, access_key, secret_key)
    request = urllib.request.Request(f"{base}{path}", headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status != 200:
                return False, f"HTTP 状态码异常：{response.status}"
            payload = json.loads(response.read().decode("utf-8"))
            doc_name = payload.get("name") or doc_id
            return True, str(doc_name)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore").strip()
        if exc.code == 401:
            return False, "认证失败 (401)：请检查 Access Key / Secret Key 是否粘贴完整且未调换位置。"
        if exc.code == 403:
            return False, "权限不足 (403)：当前密钥无权访问该文档，请确认账号与读取权限。"
        if exc.code == 404:
            return False, f"未找到文档 (404)：请确认文档 ID ({doc_id}) 是否正确。"
        return False, f"HTTP 请求失败 ({exc.code})：{detail or exc.reason}"
    except urllib.error.URLError as exc:
        return False, f"网络请求失败：{exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return False, f"解析响应失败：{exc}"


def write_credentials_file(path: Path, api_url: str, access_key: str, secret_key: str) -> None:
    """以 600 权限写入 Onshape 环境变量配置文件。"""
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(parent, 0o700)

    lines = [
        "# Onshape API 凭证（仅当前用户可读）",
        f"export ONSHAPE_API={shlex.quote(api_url)}",
        f"export ONSHAPE_ACCESS_KEY={shlex.quote(access_key)}",
        f"export ONSHAPE_SECRET_KEY={shlex.quote(secret_key)}",
        "",
    ]
    content = "\n".join(lines)

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
    finally:
        os.chmod(path, 0o600)


def _format_source_snippet(credentials_path: Path) -> str:
    """生成写入 shell 启动脚本的幂等加载语句。"""
    if credentials_path.resolve() == DEFAULT_OUTPUT_PATH.resolve():
        ref = "~/.config/onshape/credentials.env"
    else:
        ref = shlex.quote(str(credentials_path))
    return f"[ -f {ref} ] && source {ref}"


def ensure_bashrc_source(credentials_path: Path, bashrc_path: Path) -> bool:
    """幂等地将凭证加载语句追加到 bashrc，若已存在则跳过。"""
    snippet = _format_source_snippet(credentials_path)
    existing = bashrc_path.read_text(encoding="utf-8") if bashrc_path.exists() else ""
    if snippet in existing or str(credentials_path) in existing:
        return False

    addition = f"\n# 自动加载 Onshape API 环境变量\n{snippet}\n"
    if existing and not existing.endswith("\n"):
        addition = "\n" + addition
    with bashrc_path.open("a", encoding="utf-8") as handle:
        handle.write(addition)
    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="交互式配置 Onshape API Key，验证连通性并写入环境变量配置文件。"
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Onshape API 基础地址（默认：{DEFAULT_API_URL}）",
    )
    parser.add_argument(
        "--doc-id",
        default=DEFAULT_DOC_ID,
        help=f"用于校验权限的目标文档 ID（默认：{DEFAULT_DOC_ID}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"凭证输出文件路径（默认：{DEFAULT_OUTPUT_PATH}）",
    )
    parser.add_argument(
        "--no-bashrc",
        action="store_true",
        help="不自动修改 ~/.bashrc",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过在线连通性验证",
    )
    parser.add_argument(
        "--show-secret",
        action="store_true",
        help="输入 Secret Key 时在终端回显明文（默认隐藏输入）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """脚本主入口。"""
    args = _parse_args(argv)
    print("=== Onshape API Key 一键配置 ===")
    print("如尚未生成密钥，请在浏览器打开：https://cad.onshape.com/user/developer/apiKeys")
    print("（创建后请先复制 Secret Key，再复制 Access Key）\n")

    try:
        access_key = input("请粘贴 ONSHAPE_ACCESS_KEY: ").strip()
        if not access_key:
            print("错误：ONSHAPE_ACCESS_KEY 不能为空。", file=sys.stderr)
            return 1

        if args.show_secret:
            secret_key = input("请粘贴 ONSHAPE_SECRET_KEY: ").strip()
        else:
            secret_key = getpass.getpass("请粘贴 ONSHAPE_SECRET_KEY（输入时不回显，粘贴后回车）: ").strip()
        if not secret_key:
            print("错误：ONSHAPE_SECRET_KEY 不能为空。", file=sys.stderr)
            return 1
    except (KeyboardInterrupt, EOFError):
        print("\n已取消配置。")
        return 130

    if not args.no_verify:
        print(f"\n正在验证与 Onshape 文档 ({args.doc_id}) 的连通性...")
        ok, message = verify_credentials(args.api_url, args.doc_id, access_key, secret_key)
        if ok:
            print(f"✅ 验证成功！已连通 Onshape 文档：{message}")
        else:
            print(f"❌ 验证失败：{message}", file=sys.stderr)
            try:
                answer = input("是否仍要写入本地配置文件？[y/N]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\n已取消写入。")
                return 130
            if answer not in {"y", "yes"}:
                print("未写入配置文件，请核对密钥后重试。")
                return 1

    output_path = args.output.expanduser()
    write_credentials_file(output_path, args.api_url, access_key, secret_key)
    print(f"\n已安全写入凭证文件：{output_path}（权限 600）")

    if not args.no_bashrc:
        updated = ensure_bashrc_source(output_path, DEFAULT_BASHRC_PATH)
        if updated:
            print(f"已将自动加载语句追加到：{DEFAULT_BASHRC_PATH}")
        else:
            print(f"加载语句已存在于：{DEFAULT_BASHRC_PATH}（无需重复添加）")

    source_target = (
        "~/.config/onshape/credentials.env"
        if output_path.resolve() == DEFAULT_OUTPUT_PATH.resolve()
        else shlex.quote(str(output_path))
    )
    print("\n配置完成！如要在当前终端立即生效，请执行：")
    print(f"  source {source_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
