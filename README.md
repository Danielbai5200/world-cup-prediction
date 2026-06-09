# World Cup Predictor 2026

2026 世界杯预测系统。项目代码位于 [`worldcup-predictor`](./worldcup-predictor) 目录。

系统当前支持：

- 单场比赛胜平负预测
- 预期进球与最可能比分
- 0:0 至 6:6 比分概率矩阵
- 48 队世界杯模拟
- 小组出线、32 强、16 强、8 强、4 强、决赛、夺冠概率
- 中文 Streamlit 仪表盘
- 每日自动数据更新
- FIFA / Elo / OneFootball / FBref / Transfermarkt 数据源映射
- Transfermarkt 球员名单、身价和伤停数据源
- 可选赔率接口：The Odds API / Polymarket

## 本地运行

```bash
cd worldcup-predictor
pip install -r requirements.txt
python -m src.ingestion.init_database
streamlit run src/dashboard/app.py
```

浏览器访问：

```text
http://127.0.0.1:8501
```

## 数据更新

```bash
cd worldcup-predictor
python -m src.ingestion.daily_update
```

当前已接入：

- FIFA 官方排名更新时间校验
- Elo 自动抓取并写入 SQLite
- Transfermarkt 球员名单、身价、伤停标记抓取
- The Odds API 单场赔率接入，需配置 `ODDS_API_KEY`
- Polymarket 冠军市场接入，需配置 `POLYMARKET_SLUG`
- 48 支世界杯球队名单
- OneFootball URL 映射 48/48
- Transfermarkt URL 映射 48/48
- FBref URL 映射部分完成

## 详细文档

完整说明见：

[`worldcup-predictor/README.md`](./worldcup-predictor/README.md)
