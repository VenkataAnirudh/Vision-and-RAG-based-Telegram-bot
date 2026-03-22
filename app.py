import os
import struct
import sqlite3
import sqlite_vec
import torch
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
from sentence_transformers import SentenceTransformer
import ollama
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not BOT_TOKEN:
    raise ValueError(
        "No token found. Please set TELEGRAM_BOT_TOKEN in your .env file.")

# Determine if a GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Hardware computing device detected: {device.upper()}")

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

print("Loading vision model...")
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
vision_model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
).to(device)
vision_tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Define states for the ConversationHandler
ASK_STATE, IMAGE_STATE = range(2)

# Memory dictionary
user_history = {}


def update_history(user_id, role, content):
    if user_id not in user_history:
        user_history[user_id] = []
    user_history[user_id].append({"role": role, "content": content})
    if len(user_history[user_id]) > 6:
        user_history[user_id] = user_history[user_id][-6:]


def serialize_f32(vector):
    return struct.pack("%sf" % len(vector), *vector)

# --- KEYBOARD MENU CONFIGURATION ---


def get_main_menu():
    """Returns the custom keyboard layout."""
    keyboard = [['/ask', '/image', '/summarize']]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Triggers the custom keyboard and hides the typing bar."""
    await update.message.reply_text(
        "Trading Bot Online 📈\nSelect an option below:",
        reply_markup=get_main_menu()
    )
    return ConversationHandler.END


async def trigger_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Removes the custom keyboard and unlocks the standard typing bar."""
    await update.message.reply_text(
        "What is your question about trading?",
        reply_markup=ReplyKeyboardRemove()
    )
    return ASK_STATE


async def process_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text
    user_id = update.message.from_user.id

    status_msg = await update.message.reply_text("⏳ Searching documents...")

    query_bytes = serialize_f32(embedder.encode(query).tolist())

    db = sqlite3.connect("rag.db")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    cursor = db.execute('''
        SELECT documents.text_chunk, documents.source 
        FROM vec_documents 
        INNER JOIN documents ON documents.id = vec_documents.rowid
        WHERE embedding MATCH ? AND k = 2
    ''', (query_bytes,))

    results = cursor.fetchall()
    db.close()

    context_text = ""
    sources = set()
    for text_chunk, source in results:
        context_text += text_chunk + "\n\n"
        sources.add(source)

    source_string = ", ".join(sources) if sources else "None found"

    await status_msg.edit_text("🧠 Generating answer...")

    history_str = ""
    if user_id in user_history:
        for msg in user_history[user_id]:
            history_str += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""You are a concise trading assistant. Answer the user's question using ONLY the provided context.
    Keep your answer strictly under 3 sentences. Do not ramble.
    
    Context:
    {context_text}
    
    Recent Chat History:
    {history_str}
    
    Current Question: {query}
    """

    response = ollama.chat(model='phi3', messages=[
                           {'role': 'user', 'content': prompt}])
    answer = response['message']['content']

    final_reply = f"{answer}\n\n📚 **Sources:** {source_string}"

    if len(final_reply) > 4000:
        final_reply = final_reply[:4000] + "\n...[Message Truncated]"

    try:
        await status_msg.edit_text(final_reply, parse_mode='Markdown')
    except Exception:
        await status_msg.edit_text(final_reply)

    update_history(user_id, "User", query)
    update_history(user_id, "Assistant", answer)

    # Bring the menu back after answering
    await update.message.reply_text("Select your next action:", reply_markup=get_main_menu())
    return ConversationHandler.END


async def trigger_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Removes the custom keyboard so the user can use the attachment clip."""
    await update.message.reply_text(
        "Please upload the image now.",
        reply_markup=ReplyKeyboardRemove()
    )
    return IMAGE_STATE


async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    status_msg = await update.message.reply_text("📥 Downloading image...")

    photo_file = await update.message.photo[-1].get_file()
    image_path = f"temp_{user_id}.jpg"
    await photo_file.download_to_drive(image_path)

    await status_msg.edit_text("🔍 Analyzing image with Moondream2...")

    image = Image.open(image_path)
    enc_image = vision_model.encode_image(image)

    await status_msg.edit_text("📝 Generating description and extracting text...")
    caption = vision_model.answer_question(
        enc_image, "Describe this image in a short sentence.", vision_tokenizer)
    tags = vision_model.answer_question(
        enc_image, "Provide 3 short keywords for this image, separated by commas.", vision_tokenizer)

    os.remove(image_path)

    final_reply = f"This image shows {caption}\nVisible text detected: {tags}"
    await status_msg.edit_text(final_reply)

    update_history(user_id, "User", "[Uploaded an Image]")
    update_history(user_id, "Assistant", final_reply)

    # Bring the menu back after processing
    await update.message.reply_text("Select your next action:", reply_markup=get_main_menu())
    return ConversationHandler.END


async def summarize_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Directly summarizes while removing the custom keyboard."""
    user_id = update.message.from_user.id
    if user_id not in user_history or not user_history[user_id]:
        await update.message.reply_text("No history to summarize yet.", reply_markup=get_main_menu())
        return

    status_msg = await update.message.reply_text("⏳ Summarizing recent interactions...", reply_markup=ReplyKeyboardRemove())

    history_str = ""
    for msg in user_history[user_id]:
        history_str += f"{msg['role']}: {msg['content']}\n"

    prompt = f"Summarize the following brief interaction history in one short paragraph:\n{history_str}"
    response = ollama.chat(model='phi3', messages=[
                           {'role': 'user', 'content': prompt}])

    await status_msg.edit_text(f"📋 **Summary:**\n{response['message']['content']}", parse_mode='Markdown')

    # Bring the menu back
    await update.message.reply_text("Select your next action:", reply_markup=get_main_menu())


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Operation cancelled.", reply_markup=get_main_menu())
    return ConversationHandler.END


def main():
    app = Application.builder().token(BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('ask', trigger_ask),
            CommandHandler('image', trigger_image)
        ],
        states={
            ASK_STATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_question)],
            IMAGE_STATE: [MessageHandler(filters.PHOTO, process_image)]
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("summarize", summarize_chat))
    app.add_handler(conv_handler)

    print("Bot is connected to Telegram. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == '__main__':
    main()
