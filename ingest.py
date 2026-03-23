import os
import sqlite3
import sqlite_vec
import struct
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")

model = SentenceTransformer('all-MiniLM-L6-v2')

db_path = "rag.db"
if os.path.exists(db_path):
    os.remove(db_path)
    print("Deleted old rag.db file. Building fresh schema...")

db = sqlite3.connect(db_path)
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

db.execute('''
    CREATE TABLE documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text_chunk TEXT,
        source TEXT
    )
''')
db.execute('''
    CREATE VIRTUAL TABLE vec_documents USING vec0(
        embedding float[384]
    )
''')


def serialize_f32(vector):
    return struct.pack("%sf" % len(vector), *vector)


def chunk_text(text, chunk_size=600, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


data_dir = "NIFTY_50"

for filename in os.listdir(data_dir):
    if filename.endswith(".pdf"):
        filepath = os.path.join(data_dir, filename)
        reader = PdfReader(filepath)
        full_text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"

        chunks = chunk_text(full_text)
        print(f"Extracted {len(chunks)} chunks from {filename}")

        for chunk in chunks:
            cursor = db.execute(
                "INSERT INTO documents (text_chunk, source) VALUES (?, ?)", (chunk, filename))
            doc_id = cursor.lastrowid

            embedding = model.encode(chunk).tolist()
            db.execute(
                "INSERT INTO vec_documents (rowid, embedding) VALUES (?, ?)",
                (doc_id, serialize_f32(embedding))
            )

db.commit()
db.close()
print("Database generated successfully at rag.db")
