import sqlite3
from pathlib import Path

# 1. 数据库路径
db_path = Path(r"D:\Java_WorkSpace\Projects\QQbot\agent\data\knowledge.db")

# 2. 连接数据库
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 3. 查询数据
cursor.execute("SELECT * FROM messages")
rows = cursor.fetchall()

# 4. 拼字符串
lines = []
for row in rows:
    line = f"id={row['id']}, content={row['content']}, metadata={row['metadata']}, created_at={row['created_at']}"
    lines.append(line)

# 5. 输出到与数据库同目录
output_path = db_path.parent / "output.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

conn.close()
print(f"已导出 {len(lines)} 行到 {output_path}")