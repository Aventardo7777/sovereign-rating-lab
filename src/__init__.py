"""sovereign-rating-lab — 主权信用评级迁移与预测研究.

一个可复现的研究工具箱，用于：

* 构建主权信用评级面板（国家-年份-机构-评级-分值-展望-评级行动日期）
* 分析评级迁移（年度迁移矩阵、上调/下调概率、评级周期、机构差异）
* 研究宏观驱动因素（增长、通胀、债务、储备、经常账户、治理、利差）
* 建立预测模型（有序 Logit/Probit、随机森林、XGBoost、SHAP）
* 事件研究（评级行动前后利差、汇率、股指的异常变化）
* 输出可复现报告、图表、表格与 Streamlit 交互仪表盘

本包中的 ``src`` 目录名保留了研究脚手架的直观性，导入方式为::

    from src.features.rating_scale import map_rating

若需在项目外部复用，请安装为可编辑包::

    pip install -e .
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Jietao Liu"
__license__ = "MIT"

__all__ = ["__author__", "__license__", "__version__"]
