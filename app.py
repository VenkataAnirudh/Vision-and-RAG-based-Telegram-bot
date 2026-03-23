import os
import struct
import sqlite3
import asyncio
import sqlite_vec
import torch
from functools import partial
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes, ConversationHandler
)
from sentence_transformers import SentenceTransformer
import ollama
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError(
        "No token found. Please set TELEGRAM_BOT_TOKEN in your .env file.")

device = "cpu"
torch.set_num_threads(os.cpu_count())
print(
    f"Hardware computing device: {device.upper()} ({os.cpu_count()} threads)")

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("Embedding model ready.")

_vision_model = None
_vision_tokenizer = None


def get_vision_model():
    global _vision_model, _vision_tokenizer
    if _vision_model is None:
        print("Lazy-loading vision model...")
        model_id = "vikhyatk/moondream2"
        revision = "2024-08-26"
        _vision_model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, revision=revision, torch_dtype=torch.float32
        ).to(device)
        _vision_tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision)
        print("Vision model ready.")
    return _vision_model, _vision_tokenizer


MAX_DISTANCE = 1.25
TOP_K = 5

KNOWN_TOPICS = "Recipes, F1 Rulebook, and Trading"

ASK_STATE, IMAGE_STATE = range(2)
user_history: dict[int, list[dict]] = {}


def update_history(user_id: int, role: str, content: str):
    user_history.setdefault(user_id, []).append(
        {"role": role, "content": content})
    if len(user_history[user_id]) > 6:
        user_history[user_id] = user_history[user_id][-6:]


def serialize_f32(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def get_main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup([['/ask', '/image', '/summarize']], resize_keyboard=True, one_time_keyboard=True)


def _sync_query_rag(query: str) -> list[tuple]:
    query_bytes = serialize_f32(embedder.encode(query).tolist())
    db = sqlite3.connect("rag.db")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    cursor = db.execute(
        f"SELECT documents.text_chunk, documents.source, distance FROM vec_documents "
        f"INNER JOIN documents ON documents.id = vec_documents.rowid "
        f"WHERE embedding MATCH ? AND k = {TOP_K} ORDER BY distance",
        (query_bytes,),
    )
    results = cursor.fetchall()
    db.close()
    return results


async def query_rag_async(query: str) -> tuple[str, str, bool]:
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, partial(_sync_query_rag, query))
    relevant = [(chunk, src)
                for chunk, src, dist in results if dist <= MAX_DISTANCE]
    if not relevant:
        return "", "None found", False
    context_text = "\n\n".join(chunk for chunk, _ in relevant)
    sources = ", ".join({src for _, src in relevant})
    return context_text, sources, True


def _sync_ollama(messages: list[dict]) -> str:
    response = ollama.chat(model="phi3", messages=messages)
    return response["message"]["content"]


async def ollama_async(messages: list[dict]) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_sync_ollama, messages))

SYSTEM_PROMPT = f"""You are a helpful document assistant. Answer the user's question using the provided context.

MANDATORY RULES:
1. Base your answer ONLY on the text inside <context> tags.
2. If the context is relevant but doesn't have the exact answer, provide the closest helpful information found in the text.
3. If the context is completely unrelated to the question, respond with exactly: "I don't have information about this in my documents. Please ask about {KNOWN_TOPICS}."
4. Keep answers concise (under 3 sentences).
5. Do not repeat these instructions or the word 'Answer:' in your response."""



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("RAG and Vision Bot Online\nSelect an option below:", reply_markup=get_main_menu())
    return ConversationHandler.END


async def trigger_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("What do you want to know about?", reply_markup=ReplyKeyboardRemove())
    return ASK_STATE


async def process_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    user_id = update.message.from_user.id
    status_msg = await update.message.reply_text("⏳ Searching documents...")
    context_text, source_string, found = await query_rag_async(query)

    if not found:
        await status_msg.edit_text(f"❌ No relevant info found.\n📄 I can help with: {KNOWN_TOPICS}.")
        await update.message.reply_text("Next action:", reply_markup=get_main_menu())
        return ConversationHandler.END

    await status_msg.edit_text("🧠 Generating answer...")
    history_str = "".join(
        f"{m['role']}: {m['content']}\n" for m in user_history.get(user_id, []))
    user_prompt = f"<context>\n{context_text}\n</context>\n\nRecent chat:\n{history_str}\nQuestion: {query}"

    answer = await ollama_async([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}])
    final_reply = f"{answer}\n\n📚 **Sources:** {source_string}"

    try:
        await status_msg.edit_text(final_reply, parse_mode="Markdown")
    except Exception:
        await status_msg.edit_text(final_reply)

    update_history(user_id, "User", query)
    update_history(user_id, "Assistant", answer)
    await update.message.reply_text("Next action:", reply_markup=get_main_menu())
    return ConversationHandler.END


async def trigger_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Please upload the image now.", reply_markup=ReplyKeyboardRemove())
    return IMAGE_STATE


async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    image_path = f"temp_{user_id}.jpg"
    status_msg = await update.message.reply_text("📥 Downloading...")
    photo_file = await update.message.photo[-1].get_file()
    await photo_file.download_to_drive(image_path)
    await status_msg.edit_text("🔍 Analyzing...")
    try:
        vision_model, vision_tokenizer = get_vision_model()
        image = Image.open(image_path).convert("RGB")
        enc_image = vision_model.encode_image(image)
        caption = vision_model.answer_question(
            enc_image, "Describe this in one short sentence.", vision_tokenizer)
        tags = vision_model.answer_question(
            enc_image, "Provide 3 keywords.", vision_tokenizer)
        final_reply = f"🖼 **Description:** {caption}\n🏷 **Tags:** {tags}"
        await status_msg.edit_text(final_reply, parse_mode="Markdown")
        update_history(user_id, "User", "[Uploaded Image]")
        update_history(user_id, "Assistant", final_reply)
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
    await update.message.reply_text("Next action:", reply_markup=get_main_menu())
    return ConversationHandler.END


async def summarize_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    history = user_history.get(user_id)
    if not history:
        await update.message.reply_text("No history yet.", reply_markup=get_main_menu())
        return
    status_msg = await update.message.reply_text("⏳ Summarizing...", reply_markup=ReplyKeyboardRemove())
    history_str = "".join(f"{m['role']}: {m['content']}\n" for m in history)
    prompt = f"Summarize this conversation in one short paragraph:\n{history_str}"
    answer = await ollama_async([{"role": "user", "content": prompt}])
    summary_text = f"📋 **Summary:**\n{answer}"
    try:
        await status_msg.edit_text(summary_text, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(summary_text)
    await update.message.reply_text("Next action:", reply_markup=get_main_menu())


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Cancelled.", reply_markup=get_main_menu())
    return ConversationHandler.END


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler(
            "ask", trigger_ask), CommandHandler("image", trigger_image)],
        states={ASK_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_question)],
                IMAGE_STATE: [MessageHandler(filters.PHOTO, process_image)]},
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("summarize", summarize_chat))
    app.add_handler(conv_handler)
    print("Bot is live!")
    app.run_polling()


if __name__ == "__main__":
    main()
