"""快速导入验证。用完即删。"""
import sys
sys.path.insert(0, '.')
from app.config import get_settings
s = get_settings()
print('DEEPSEEK_API_KEY:', repr(s.deepseek_api_key[:8]) if s.deepseek_api_key else 'empty')
print('EMBEDDING_API_KEY:', repr(s.embedding_api_key[:8]) if s.embedding_api_key else 'empty')

from app.database import MessageStore
ms = MessageStore(':memory:')
print('MessageStore OK')

from app.llm import build_embeddings
emb = build_embeddings(s)
print('build_embeddings:', 'None (as expected)' if emb is None else 'got instance')

from app.knowledge import KnowledgeBase
print('KnowledgeBase class OK')

from app.main import app
print('App import OK')