# Multimodal RAG + Vision Telegram Bot

A highly efficient Telegram bot that answers questions from your own PDF documents locally and analyses uploaded images using advanced Vision APIs with seamless local fallbacks.

> Built with: `sqlite-vec` · `sentence-transformers` · `Ollama / phi3` · `Featherless API / Qwen` · `python-telegram-bot`
---

## Table of Contents

1. [What It Does](#what-it-does)
2. [System Architecture](#system-architecture)
3. [Prerequisites](#prerequisites)
4. [Project Structure](#project-structure)
5. [Setup - Step by Step](#setup---step-by-step)
6. [Commands & Usage](#commands--usage)
7. [Critical Catches](#critical-catches)
8. [Troubleshooting](#troubleshooting)

---

## What It Does

| Feature | How it works |
|---|---|
| `/ask` | Searches your PDF documents using vector similarity, then answers using only what's in those documents (RAG). Off-topic questions are blocked. |
| `/image` | Accepts a photo and returns a description + 3 keyword tags using the ultra-fast Featherless API (`Qwen/Qwen3.6-27B`). If the API is missing or fails, it flawlessly falls back to a local Ollama `moondream` engine. |
| `/summarize` | Summarises the last 6 turns of your conversation using phi3 via Ollama. |

**Documents currently loaded** (from the `Documents/` folder):
- `The-Complete-Guide-to-Trading.pdf`
- `F1_Rulebook.pdf`
- `Recipes.pdf` (Manjula's Kitchen - Indian Vegetarian)

---

## System Architecture

```
Telegram User
      │
      ▼
python-telegram-bot (app.py)
      │
      ├─── /ask ──► sentence-transformers (embed query)
      │                     │
      │             sqlite-vec (vector search → rag.db)
      │                     │
      │             Ollama / phi3 (answer from context only)
      │
      ├─── /image ──► Featherless API (Qwen/Qwen3.6-27B) [Primary]
      │                     │
      │               Ollama / moondream [Fallback]
      │
      └─── /summarize ──► Ollama / phi3
```

**ingest.py** (Run via `run.bat` or manually to build the database):
```
Documents/*.pdf
      │
      ▼
PyPDF2 (extract text)
      │
      ▼
sentence-transformers / all-MiniLM-L6-v2 (embed chunks)
      │
      ▼
sqlite-vec → rag.db
```

---

## Prerequisites

### 1. Python 3.10 or higher
Download from [python.org](https://www.python.org/downloads/).  
Verify: `python --version`

### 2. Ollama
Ollama serves the local models (`phi3` and `moondream`).
1. Download from [ollama.com](https://ollama.com/) and install it.
2. Ensure Ollama is **running in the background**. On Windows, it runs as a system tray icon.

### 3. A Telegram Bot Token
1. Open Telegram and message `@BotFather`.
2. Send `/newbot` and follow the prompts.
3. Copy the token.

### 4. Featherless API Key (Optional but Recommended)
1. Get an API key from Featherless.ai. This powers the high-speed vision generation.

---

## Project Structure

```
Doc Image Helper/
├── Documents/                  ← Put your PDF files here (required before ingest)
├── venv/                       ← Python virtual environment (auto-created by run.bat)
├── app.py                      ← Main bot codebase
├── ingest.py                   ← PDF ingestion script
├── rag.db                      ← Auto-generated vector database (do not edit)
├── requirements.txt            ← All Python dependencies
├── run.bat                     ← Automated end-to-end launcher script
├── .env                        ← Your API Keys (created during setup)
└── .gitignore
```

---

## Setup - Step by Step

### Step 1 - Clone the repository

```bash
git clone <repository-url>
cd "Doc Image Helper"
```

### Step 2 - Create your `.env` file

Create a file named `.env` in the project root with the following content:

```env
TELEGRAM_BOT_TOKEN=paste_your_bot_token_here
FEATHERLESS_API_KEY=paste_your_featherless_key_here
```
*(If you leave `FEATHERLESS_API_KEY` blank, the bot will automatically fall back to running the vision processing entirely locally through Ollama).*

### Step 3 - Add your documents

1. Create the `Documents/` folder inside the project root if it does not exist.
2. Copy your PDF files into the `Documents/` folder.

### Step 4 - Run the Bot (Automated Setup)

Simply double-click the **`run.bat`** file in Windows.

The script acts as an automated end-to-end launcher that will:
1. Automatically create your Python `venv`.
2. Install all required dependencies from `requirements.txt`.
3. Check and pull the required Ollama models (`phi3` and `moondream`).
4. Prompt you (Y/N) to ingest the PDF documents to build `rag.db`.
5. Safely start the bot.

Once you see `Bot is live!` in the console, open Telegram and message it `/start`.

---

## Commands & Usage

| Command | What to do |
|---|---|
| `/start` | Displays the main menu with buttons. |
| `/ask` | Bot asks for your question. Type it and send. The bot searches the loaded PDFs and answers using only what is in the documents. Off-topic questions are rejected. |
| `/image` | Bot asks you to upload an image. **Send it as a regular compressed photo** (tap the photo icon → select from gallery → tap Send). Do NOT send it as a File/Document. |
| `/summarize` | Summarises the last 6 turns of your conversation. |
| `/cancel` | Cancels the current operation and returns to the main menu. |

---

## Critical Catches

### Ollama must be running before starting
If the bot starts but `/ask` or `/image` fallbacks fail, check that Ollama is running in your Windows System Tray. The `run.bat` script handles model installation automatically.

### Image must be sent as a compressed photo, not a file
When using `/image`, you **must** send the image as a standard Telegram photo:
- ✅ Tap the 📎 attachment icon → **Gallery/Photos** → select → **Send**
- ❌ Do NOT tap **File** - Telegram will send it as a Document, which bypasses photo handling and causes a download error

### rag.db is stale after adding new PDFs
If you add a new PDF to `Documents/`, you must rebuild the database. Simply restart `run.bat` and press `Y` when it prompts you to re-ingest your documents.

### Bot does not answer general knowledge questions
This is intentional. The RAG pipeline rejects queries where the vector similarity distance is too high (no matching content in your documents). If a legitimate question from your documents is being rejected, you can raise `MAX_DISTANCE` in `app.py` (default: `1.23`).

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ValueError: No token found` | `.env` file missing or token not set | Create `.env` with `TELEGRAM_BOT_TOKEN=your_token` |
| `sqlite3.OperationalError` | Incompatible sqlite-vec version | The `requirements.txt` is pinned to `>=0.1.9` which fixes Windows DLL bugs. Re-run `run.bat` to update dependencies. |
| `model requires more system memory... (50.0 GiB)` | Ollama allocating huge context window | `app.py` limits context explicitly (`num_ctx: 4096`). Make sure you are using the updated `app.py`. |
| `ConnectionError` on `/ask` | Ollama is not running | Start Ollama, verify with `ollama list` |
| `No relevant info found` | `MAX_DISTANCE` threshold too strict | Re-run `run.bat` and ingest PDFs; or raise `MAX_DISTANCE` in `app.py` |
