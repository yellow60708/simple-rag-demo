\# Thesis Semantic RAG Pipeline



基於 ChromaDB 與開源中文 Embedding 模型打造的輕量化檢索增強生成（RAG）系統 Demo。



\## 專案概述

本專案實作標準 RAG 架構中的前段核心檢索流水線（Document -> Chunking -> Vector Embedding -> Vector Database -> Semantic Retrieval -> Prompt Assembly）。

以碩士論文研究成果（零樣本語音情緒識別與端到端自動化 Pipeline）作為檢索知識源，示範如何透過語意向量相似度（Cosine Distance）精準檢索出相關實驗與系統背景段落。



\## 特色與技術亮點

\- \*\*架構設計\*\*：完整實作文件切塊（Chunking）、向量特徵儲存、Top-K 相似度比對與上下文 Prompt 組裝。

\- \*\*本地開源生態\*\*：採用 Hugging Face 開源中文語意模型 (`shibing624/text2vec-base-chinese`)，免付費 API Key，具備地端推論與資料隱私優勢。

\- \*\*互動式檢索\*\*：提供 CLI 終端機自然語言即時問答與向量距離評估。



\## 技術棧 (Tech Stack)

\- \*\*Language\*\*: Python 3.x

\- \*\*Vector Database\*\*: ChromaDB

\- \*\*Embedding Model\*\*: Sentence-Transformers (`shibing624/text2vec-base-chinese`)



\## 快速啟動

```bash

pip install -r requirements.txt

python app.py

<img width="1102" height="620" alt="image" src="https://github.com/user-attachments/assets/0568dd9b-e1a3-4e59-93f8-7d7260f589b0" />


