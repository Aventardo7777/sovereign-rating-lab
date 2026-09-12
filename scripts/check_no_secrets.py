"""pre-commit 钩子：阻止把硬编码密钥提交到仓库。

检查对象是 ``* .py / .yaml / .yml / .toml / .json`` 文件。命中即返回非零退出码。

用法
----
.. code-block:: bash

    python scripts/check_no_secrets.py path/to/file1 path/to/file2

检测模式覆盖常见凭证形态：``api_key = "...."``、``token: xxxx``、
AWS Access Key ID、私钥块、Bearer 令牌等。另外会放过明显是占位符的值
（``your_key_here``、``xxx``、``<...>``、空字符串、``None``）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# 允许的占位符（大小写不敏感，匹配即视为安全）
PLACEHOLDER = re.compile(
    r"^(?:$|none|null|changeme|change_me|your[_-]?\w*|example\w*|placeholder|"
    r"xxx+|\*{3,}|<[^>]*>|\$\{[^}]*\}|\{\{[^}]*\}\}|\.\.\.|todo|test|dummy|fake|sample)",
    re.IGNORECASE,
)

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        # 只匹配**带引号**的赋值，避免把 `api_key = get_env(env_name)` 这类
        # 正确写法的函数调用误判为硬编码密钥。
        "疑似硬编码凭证（带引号的字面值）",
        re.compile(r"""(?ix)
            (?:api[_-]?key|apikey|secret[_-]?key|access[_-]?key|client[_-]?secret|
               auth[_-]?token|bearer[_-]?token|password|passwd|credential)
            \s*[:=]\s*
            ["']([^"'\s,#}{]{8,})["']
            """),
    ),
    (
        "AWS Access Key ID",
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ),
    (
        "私钥块",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP |DSA )?PRIVATE KEY-----"),
    ),
    (
        "GitHub Token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    ),
    (
        "Slack Token",
        re.compile(r"\bxox[abpsr]-[A-Za-z0-9-]{10,}\b"),
    ),
    (
        "通用长十六进制密钥",
        re.compile(r"""(?i)\b(?:secret|token)\b\s*[:=]\s*["']?[0-9a-f]{32,}["']?"""),
    ),
]

ALLOW_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json"}

#: 这些路径下的文件不参与扫描（测试夹具与钩子脚本本身会包含规则字样）
SKIP_PREFIXES = ("tests/", "tests\\")
SKIP_EXACT = {"scripts/check_no_secrets.py"}


def scan(path: Path) -> list[tuple[int, str, str]]:
    """扫描单个文件，返回 ``(行号, 规则名, 片段)`` 列表。"""
    findings: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return findings

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        is_comment = stripped.startswith(("#", "//", "*")) and "=" not in stripped.split(":")[0]
        # 注释行一般跳过，但仍要检查私钥块（可能被注释掉的密钥）
        if is_comment and "PRIVATE KEY" not in line:
            continue
        for name, pattern in PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            captured = match.group(1) if match.groups() else match.group(0)
            if PLACEHOLDER.match(str(captured).strip().strip("\"'")):
                continue
            findings.append((line_number, name, line.strip()[:120]))
            break
    return findings


def main(argv: list[str]) -> int:
    targets = [Path(arg) for arg in argv[1:]]
    problems = 0
    for path in targets:
        if not path.is_file() or path.suffix.lower() not in ALLOW_SUFFIXES:
            continue
        normalized = str(path).replace("\\", "/")
        if normalized in SKIP_EXACT or normalized.startswith(SKIP_PREFIXES):
            continue
        for line_number, rule, snippet in scan(path):
            problems += 1
            print(f"{path}:{line_number}: [{rule}] {snippet}")

    if problems:
        print(
            f"\n❌ 检测到 {problems} 处疑似硬编码凭证。\n"
            '   请改用环境变量：在 src/config.py 中用 get_env("YOUR_ENV_VAR") 读取，\n'
            "   并把变量名写入 .env.example（不要写入真实值）。\n"
            "   若确认是误报（例如占位符或测试数据），请调整脚本中的 PLACEHOLDER 规则。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
