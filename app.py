import chromadb
from chromadb.utils import embedding_functions

# 1. 初始化 Chroma 記憶體客戶端
client = chromadb.Client()

# 2. 換用強大的開源中文專用 Embedding 模型（第一次跑會自動下載，約幾十 MB）
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="shibing624/text2vec-base-chinese"
)

# 3. 建立或重置 Collection
collection_name = "thesis_knowledge_base"
try:
    client.delete_collection(name=collection_name)
except Exception:
    pass

collection = client.create_collection(
    name=collection_name,
    embedding_function=embedding_func
)

# 4. 讀取知識庫檔案 (thesis_data.txt)
with open("thesis_data.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f.readlines() if line.strip()]

# 5. 寫入向量資料庫
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print("=" * 60)
print(f"✅ 成功載入 {len(chunks)} 筆論文區塊，已啟用中文專用語意向量模型！")
print("💡 輸入問題進行檢索；輸入 'q' 或 'exit' 退出。")
print("=" * 60 + "\n")

# 6. 互動查詢
def ask():
    while True:
        try:
            query = input("💬 請輸入你的問題: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "q", "quit"]:
                print("👋 程式結束！")
                break

            # 檢索前 2 筆最相關資料
            results = collection.query(query_texts=[query], n_results=2)
            docs = results["documents"][0]
            distances = results["distances"][0]

            print("\n🔍 【RAG 檢索結果】相關內容段落：")
            for i, (doc, dist) in enumerate(zip(docs, distances), 1):
                # 距離越小代表語意越相近
                print(f"  [{i}] {doc} (差異距離: {dist:.4f})")

            # 組裝 Prompt 預覽
            context = "\n".join(docs)
            print("\n🤖 【組裝給 LLM 的上下文 Prompt】:")
            print(f"> 參考上下文：\n{context}\n> 問題：{query}\n")
            print("-" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\n👋 程式結束！")
            break

if __name__ == "__main__":
    ask()