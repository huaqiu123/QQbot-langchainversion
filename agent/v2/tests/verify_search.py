"""直接测知识库检索"""
import sys, warnings, traceback
warnings.filterwarnings("ignore")
sys.path.insert(0, r"d:\Java_WorkSpace\Projects\QQbot\agent")

out = open(r"d:\Java_WorkSpace\Projects\QQbot\agent\verify_out.txt", "w", encoding="utf-8")
try:
    from app.config import get_settings
    from app.llm import build_embeddings
    from app.knowledge import KnowledgeBase

    s = get_settings()
    out.write(f"threshold={s.knowledge_similarity_threshold}\n")
    emb = build_embeddings(s)
    out.write(f"emb={emb is not None}\n")
    kb = KnowledgeBase(s.chroma_persist_dir, emb, similarity_threshold=0.0, search_top_k=10)
    out.write(f"chunks={kb.get_chunk_count()}\n")
    out.write(f"available={kb.available}\n")

    for q in ["邱伟华", "武汉大学"]:
        out.write(f"\n查询: {q}\n")
        hits = kb.search(q)
        out.write(f"  hits={len(hits)}\n")
        for r in hits:
            out.write(f"  s={r['similarity']:.4f} | {r['content'][:80]}\n")
except:
    out.write(traceback.format_exc())
out.close()