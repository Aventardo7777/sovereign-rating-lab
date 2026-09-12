"""pre-commit 钩子：阻止把真实数据或抓取缓存提交到仓库。

本项目允许纳入版本控制的只有：

* ``data/sample/*.csv`` —— 人工生成的合成示例数据
* ``data/raw/ratings_template.csv`` —— 空的导入模板
* ``data/raw/README.md``、各目录下的 ``.gitkeep``

其余 ``data/raw``、``data/interim``、``data/processed`` 下的文件一律拒绝提交，
原因有二：

1. 其中可能包含第三方评级机构的**版权数据**，提交即构成侵权风险；
2. 抓取缓存（``data/raw/cache/``）包含来自目标站点的原始内容。

用法
----
.. code-block:: bash

    python scripts/check_no_raw_data.py data/raw/ratings.csv
"""

from __future__ import annotations

import sys
from pathlib import PurePosixPath

ALLOWED = {
    "data/raw/ratings_template.csv",
    "data/raw/README.md",
    "data/raw/.gitkeep",
    "data/interim/.gitkeep",
    "data/processed/.gitkeep",
}

ALLOWED_PREFIXES = (
    "data/sample/",
    "docs/",
    "notebooks/",
)


def main(argv: list[str]) -> int:
    blocked: list[str] = []
    for arg in argv[1:]:
        normalized = PurePosixPath(str(arg).replace("\\", "/")).as_posix()
        if not normalized.startswith("data/"):
            continue
        if normalized in ALLOWED or normalized.startswith(ALLOWED_PREFIXES):
            continue
        blocked.append(normalized)

    if blocked:
        print("❌ 以下数据文件不允许提交到 Git 仓库：")
        for path in blocked:
            print(f"   - {path}")
        print(
            "\n   数据目录的用途：\n"
            "   * data/raw/       用户导入的原始数据（请勿提交）\n"
            "   * data/interim/   清洗中间产物（可由流水线重新生成）\n"
            "   * data/processed/ 建模用面板（可由流水线重新生成）\n"
            "   * data/sample/    合成演示数据（已在版本控制中）\n"
            "\n   若确实需要共享数据，请提供下载脚本或生成脚本，而非直接提交数据文件。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
