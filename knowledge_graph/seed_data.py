"""
内置种子金融知识库。

用途：给知识图谱提供基础的关系数据（常见A股公司/行业/股票/供应链/竞争/人事关系），
让"关系推理"不依赖舆情数据本身；舆情记录会在此基础上自动补充 公司-事件、公司共现 关系。

本文件是公司/行业/股票代码等基础信息的唯一数据源：market_data/provider.py 的
代码解析、repository 的行业热度榜等都从这里引用，不各自维护映射表。

数据刻意做小（几十个实体），便于维护和演示；生产环境可按需扩充或从外部数据源导入。
"""
# 公司注册表：canonical -> {aliases, industry, stock}
COMPANIES = {
    "贵州茅台": {"aliases": ["贵州茅台", "茅台", "600519"], "industry": "白酒", "stock": "600519.SH"},
    "五粮液": {"aliases": ["五粮液"], "industry": "白酒", "stock": "000858.SZ"},
    "宁德时代": {"aliases": ["宁德时代", "宁王", "CATL"], "industry": "新能源", "stock": "300750.SZ"},
    "比亚迪": {"aliases": ["比亚迪", "BYD"], "industry": "汽车", "stock": "002594.SZ"},
    "特斯拉": {"aliases": ["特斯拉", "Tesla"], "industry": "汽车", "stock": "TSLA"},
    "招商银行": {"aliases": ["招商银行", "招行"], "industry": "银行", "stock": "600036.SH"},
    "工商银行": {"aliases": ["工商银行", "工行"], "industry": "银行", "stock": "601398.SH"},
    "中国平安": {"aliases": ["中国平安", "平安"], "industry": "保险", "stock": "601318.SH"},
    "中信证券": {"aliases": ["中信证券"], "industry": "券商", "stock": "600030.SH"},
    "东方财富": {"aliases": ["东方财富"], "industry": "券商", "stock": "300059.SZ"},
    "隆基绿能": {"aliases": ["隆基绿能", "隆基股份"], "industry": "光伏", "stock": "601012.SH"},
    "迈瑞医疗": {"aliases": ["迈瑞医疗"], "industry": "医药", "stock": "300760.SZ"},
    "恒瑞医药": {"aliases": ["恒瑞医药"], "industry": "医药", "stock": "600276.SH"},
}

# 行业清单（用于"有哪些X"这类查询的兜底识别）
INDUSTRIES = ["白酒", "新能源", "汽车", "银行", "保险", "券商", "光伏", "医药"]

# 查不到所属行业时的兜底类目（行业热度榜等跨公司对比依赖此归一化）
DEFAULT_INDUSTRY = "其他行业"


def resolve_industry(company_or_alias: str) -> str:
    """
    公司名/别名 -> 所属行业。和 market_data/provider.py 的 resolve_stock_code()
    用的是同一份 COMPANIES 别名表，保证"这家公司叫什么/属于哪个行业/股票代码是什么"
    三处查询结果互相一致，不会出现同一家公司在不同模块里对应到不同行业的情况。
    查不到时归到"其他行业"兜底类目，不返回None——行业热度榜按行业GROUP BY时
    不用再额外处理空值。
    """
    if not company_or_alias:
        return DEFAULT_INDUSTRY
    for canonical, info in COMPANIES.items():
        if company_or_alias == canonical or company_or_alias in info.get("aliases", []):
            return info["industry"]
    return DEFAULT_INDUSTRY


def find_known_company_in_text(text: str) -> str | None:
    """
    在一段文本里扫描是否提到了种子库内的已知公司（任意别名命中即可），返回canonical公司名；
    没提到任何已知公司则返回None。

    轻量模型快速通道用到：轻量模型不做通用命名实体识别，
    只能可靠识别"种子库里已经登记过的公司"，遇到没登记过的公司名一律
    交给LLM处理（LLM的NER能力覆盖任意公司，不受限于这张小词表）。
    """
    if not text:
        return None
    for canonical, info in COMPANIES.items():
        for alias in info.get("aliases", []):
            if alias and alias in text:
                return canonical
    return None

# 人员注册表：person -> {aliases}
PEOPLE = {
    "曾毓群": {"aliases": ["曾毓群"]},
    "王传福": {"aliases": ["王传福"]},
    "马斯克": {"aliases": ["马斯克", "Elon Musk"]},
}

# 关系三元组：(src, relation, dst, props)；src/dst 若是公司名/人名会按注册表归一化到对应节点
SEED_RELATIONS = [
    # 供应链（电池供应整车厂）
    ("宁德时代", "supplies", "特斯拉", {}),
    ("宁德时代", "supplies", "比亚迪", {}),
    # 竞争关系
    ("比亚迪", "competes_with", "特斯拉", {}),
    ("宁德时代", "competes_with", "隆基绿能", {}),
    ("中信证券", "competes_with", "东方财富", {}),
    ("五粮液", "competes_with", "贵州茅台", {}),
    # 人事（高管任职）
    ("曾毓群", "works_at", "宁德时代", {"role": "董事长"}),
    ("王传福", "works_at", "比亚迪", {"role": "董事长"}),
    ("马斯克", "works_at", "特斯拉", {"role": "CEO"}),
]
