import telebot
import pandas as pd
import numpy as np
import sys
import os
import torch
import gc
import threading
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer

# --- Конфигурация: Токен бота ТГ, используемый датасет, имя файла с кэшем для sbert_large_nlu_ru
API_TOKEN = '' # Вставьте сюда токен вашего бота
DATA_FILE_NAME = 'faq_dataset.csv'
VECTORS_FILE_NAME = 'faq_vectors.npy'

# Очистка памяти перед стартом
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# Определение устройства
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Устройство для вычислений: {device.upper()}")

# Подгтовка файлов и путей
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, DATA_FILE_NAME)
VECTORS_FILE = os.path.join(SCRIPT_DIR, VECTORS_FILE_NAME)

print("1. Загрузка модели поиска (SBERT)...")
search_model = SentenceTransformer('sberbank-ai/sbert_large_nlu_ru')

if not os.path.exists(DATA_FILE):
    print(f"[ERROR] Файл '{DATA_FILE_NAME}' не найден!")
    sys.exit(1)

try:
    df = pd.read_csv(DATA_FILE, sep=';')
    df.columns = map(str.lower, df.columns)
    if 'question' not in df.columns or 'answer' not in df.columns:
        print("[ERROR] ОШИБКА: В CSV нет колонок 'question' и 'answer'.")
        sys.exit(1)
    print(f"[OK] Загружено {len(df)} пар вопрос-ответ.")
except Exception as e:
    print(f"[ERROR] ОШИБКА CSV: {e}")
    sys.exit(1)

# Векторизация с кэшированием базы вопросов
if os.path.exists(VECTORS_FILE):
    print("[INFO] Загрузка векторов...")
    faq_vectors = np.load(VECTORS_FILE)
    if len(faq_vectors) != len(df):
        print("[WARNING] База обновилась, пересчитываем векторы...")
        faq_vectors = search_model.encode(df['question'].tolist(), show_progress_bar=True)
        np.save(VECTORS_FILE, faq_vectors)
else:
    print("2. Векторизация базы...")
    faq_vectors = search_model.encode(df['question'].tolist(), show_progress_bar=True)
    np.save(VECTORS_FILE, faq_vectors)

# Легкая модель Qwen 2.5 3B (~6 GB VRAM)
HF_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# Загрузка LLM
print(f"3. Загрузка LLM '{HF_MODEL_NAME}'...")

try:
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        dtype = torch.float16,
        device_map=device
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    print("[OK] LLM готова к работе!")

except Exception as e:
    print(f"[ERROR] ОШИБКА ЗАГРУЗКИ LLM: {e}")
    sys.exit(1)

# Функция обновления состояния "Печатает" в боте
def keep_typing(bot_instance, chat_id, stop_event):
    while not stop_event.is_set():
        try:
            bot_instance.send_chat_action(chat_id, 'typing')
        except Exception as e:
            print(f"[WARNING] Ошибка отправки состояния 'Печатает': {e}")
        time.sleep(4)

# Функция генерации ответа
def generate_rag_answer(user_query, context_list):
    full_context = "\n---\n".join(context_list)
    
    system_instruction = (
        "Ты — ассистент магистратуры, который цитирует документы ДОСЛОВНО и ПОЛНОСТЬЮ.\n"
        "Твоя цель — выдать пользователю ВСЮ информацию, найденную в контексте, ничего не упуская.\n"
        "Не останавливайся на половине. Если в контексте есть несколько списков или разделов — выведи их все.\n"
        "Если информация не найдена, просто скажи: 'Извините, у меня нет информации по этому вопросу, обратитесь к координатору.'\n"
        "Не используй слово 'контекст'.\n"
        "Не сокращай текст. Отвечай на русском языке."
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Контекст:\n{full_context}\n\nВопрос: {user_query}"}
    ]

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    MAX_NEW_TOKENS = 768 
    TEMPERATURE = 0.1
    generation_kwargs = dict(
        text_inputs=messages,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
        top_p=0.9,               # Отсекаем совсем маловероятное
        repetition_penalty=1.15, # Штраф за повторы  
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  
        streamer=streamer
    )

    thread = threading.Thread(target=generator, kwargs=generation_kwargs)
    thread.start()

    print(f"\n[INFO] Генерация (Qwen 3B)...")
    generated_text = ""
    
    with tqdm(total=MAX_NEW_TOKENS, unit="tok", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        for new_text in streamer:
            generated_text += new_text
            pbar.update(1)
            
    return generated_text.strip()

# Бот Telegram
bot = telebot.TeleBot(API_TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Здравствуйте! Я виртуальный помощник для студентов и обучающихся " \
    "по программе магистратуры УрФУ 'Прикладной искусственный интеллект'.\n\n" \
    "Все ответы даются исключительно в ознакомительных целях.")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_query = message.text
    print(f"\n[QUERY] Запрос: {user_query}")

    # 1. Поиск контекста
    query_vector = search_model.encode([user_query])
    similarities = cosine_similarity(query_vector, faq_vectors)[0]
    
    top_k = 5
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    found_contexts = []
    
    THRESHOLD = 0.5
    print("\n[SEARCH] Результаты поиска (SBERT):")
    for i, idx in enumerate(top_indices):
        score = similarities[idx]
        if score > THRESHOLD:
            answer_text = df.iloc[idx]['answer']
            found_contexts.append(answer_text)
            
            # Отладочный вывод контекста
            preview = (answer_text[:100] + '...') if len(answer_text) > 100 else answer_text
            print(f"[{i+1}] Score: {score:.4f} | {preview}")
        else:
            # Можно раскомментировать, чтобы увидеть, что было отсеяно
            # print(f"[{i+1}] Score: {score:.4f} (Ниже порога)")
            pass

    if found_contexts:
        # Создаем событие для остановки статуса "Печатает"
        stop_typing_event = threading.Event()
        
        # Запускаем поток для поддержания статуса "Печатает"
        typing_thread = threading.Thread(
            target=keep_typing, 
            args=(bot, message.chat.id, stop_typing_event),
            daemon=True
        )
        typing_thread.start()
        
        # 2. Фоновая генерация
        def background_process():
            try:
                final_answer = generate_rag_answer(user_query, found_contexts)
                
                # Останавливаем обновление состояния "Печатает"
                stop_typing_event.set()
                typing_thread.join(timeout=1)
                
                bot.reply_to(message, final_answer)
                print("[OK] Отправлено.")
            except Exception as e:
                stop_typing_event.set()
                print(f"[ERROR] Ошибка: {e}")
                bot.reply_to(message, "Произошла ошибка при генерации ответа.")

        threading.Thread(target=background_process, daemon=True).start()
    else:
        print("[WARNING] Нет релевантного контекста.")
        bot.reply_to(message, "Извините, в базе знаний нет информации по этому вопросу.")

if __name__ == '__main__':
    print("[START] Бот запущен...")
    bot.infinity_polling()
