import sqlite3
db = sqlite3.connect('rag.db')
db.execute("DELETE FROM categories WHERE name='三菱'")
db.commit()
print('done')
db.close()