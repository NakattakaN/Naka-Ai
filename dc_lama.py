import discord
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import asyncio
from PIL import Image
import os
import sqlite3
import time
from datetime import datetime
import faiss
import mss
import cv2
from faster_whisper import WhisperModel
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import nest_asyncio
from collections import defaultdict
import logging
import random
from na_vision import SmolVLM2
from na_tts import TTTS
import keyboard
from na_stt import FastWhisperRecognizer
import accelerate
accelerate.utils.modeling.get_balanced_memory

last_bot_reply = None
input_lock = asyncio.Lock()
i=1
  # or filename string
def bot(discordd = bool,screen = bool,voice= bool,talk_local = bool,repeat = bool):

    nest_asyncio.apply()
    logging.basicConfig(
        filename="console_log.txt",    # file to write
        filemode="w",                  # overwrite each run; use "a" to append
        level=logging.INFO,            # capture INFO and above; change to DEBUG for more
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Bot starting up…")

    if screen == True:
        smolvlm = SmolVLM2()
    

    def capture_screen():
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # full screen
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        

    if voice == True:
        ttts = TTTS()

    if talk_local == True:
        recognizer = FastWhisperRecognizer()

    DISCORD_TOKEN = "xxxxxx"

    # === Model Paths ===
    BASE_MODEL = r"base model path"
    LORA_DIR = r"lora path"


    #database config
    DB_PATH       = r"C:\Users\atoca\Desktop\Naka-chan\local model\Naka-Brain\Memories\memo.db"
    FAISS_INDEX   = "memory_index.faiss"
    EMBED_DIM     = 384               
    TOP_K         = 3
    MODEL_NAME    = r"C:\Users\atoca\Desktop\Naka-chan\local model\Naka-Brain\Memories\hipocam"  


    # 1) SQLite for metadata
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        user_nick TEXT,
        text       TEXT
    )
    """)

    conn.commit()

    # 2) Sentence-Transformer for embeddings
    embedder = SentenceTransformer(MODEL_NAME)


    # 3) FAISS index
    if os.path.exists(FAISS_INDEX):
        index = faiss.read_index(FAISS_INDEX)
    else:
        # flat index for simplicity
        index = faiss.IndexFlatL2(EMBED_DIM)

    def save_index():
        faiss.write_index(index, FAISS_INDEX)

    # ── SELECTIVE STORAGE RULE ────────────────────────────────────────────────────

    def should_store(text: str) -> bool:
        keywords = ["fair",  "ok",  "i see",  "got it",  "cool",  "nice",  "thanks",  "sure",  "maybe",  "alright",  "no way",  "seriously?",  "whatever",  "exactly",  "relax",  "hang on",  "my bad",  "no problem",  "i like",  "i love",  "remember that",  "good point",  "makes sense",  "not bad",  "so far",  "by the way",  "just kidding",  "no worries",  "good one",  "well done", "fuck"  ]
        user_part = text.split("Naka:")[0].strip() 
        user_input = user_part.replace("User:", "").strip()  
        first_word = user_input.split()[0].lower() if user_input else ""
        if first_word in keywords:
            return None
        return text 


    # ── MEMORY OPERATIONS ─────────────────────────────────────────────────────────
    c.execute("SELECT id FROM memories ORDER BY id")
    memory_ids = [row[0] for row in c.fetchall()]

    def rebuild_index():
        index.reset()
        memory_ids.clear()
        c.execute("SELECT id, text FROM memories")
        for rowid, text in c.fetchall():
            emb = embedder.encode([text]).astype("float32")
            index.add(emb)
            memory_ids.append(rowid)
        save_index()


    def store_memory(user_nick: str, text: str):
        if not should_store(text):
            return
        ts = datetime.utcnow().isoformat()
        c.execute(
            "INSERT INTO memories (timestamp, user_nick, text) VALUES (?, ?, ?)",
            (ts, user_nick, text)
        )
        rowid = c.lastrowid
        conn.commit()

        # encode & add to FAISS
        emb = embedder.encode([text]).astype("float32")
        index.add(emb)

        # keep our python mapping in sync
        memory_ids.append(rowid)

        save_index()


    def fetch_memories(query: str, user_nick: str, k: int = TOP_K):
        # 1) embed query and search FAISS
        q_emb = embedder.encode([query]).astype("float32")
        D, I  = index.search(q_emb, k*20)  # search a few extra so we can filter
        results = []

        for idx in I[0] - MAX_HISTORY_LENGTH:
            if idx < 0 or len(results) >= k:
                break
            rowid = memory_ids[idx]
            c.execute("SELECT user_nick, text FROM memories WHERE id = ?", (rowid,))
            r = c.fetchone()
            if not r:
                continue
            nick, text = r
            #if nick != user_nick:
                #continue
            results.append(f"Old Conversation with  {nick} = {text}")

        return results

    # === Conversation History ===
    conversation_history = defaultdict(list)  # channel_id -> list of (role, content)
    MAX_HISTORY_LENGTH = 3  # max number of turns (user+assistant)




    # === Load Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Quantization Config ===
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # === Load Base + LoRA ===
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        LORA_DIR,
        device_map="cuda",
        torch_dtype=torch.float16,
    )
    print("Model loaded on device:", model.base_model.model.model.device)
    model.eval()


    # === Discord Client ===
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    def clean_response(response: str) -> str:
        # Keep only first line and clean formatting
        first_line = response.strip().split("\n")[0]
        first_line.replace("**", "*")
        cleaned = EMOJI_PATTERN.sub("", first_line)
        for pat, repl in WORD_SUBS.items():
            cleaned = re.sub(pat, repl, cleaned, flags=re.IGNORECASE)
        return cleaned
    


    EMOJI_PATTERN = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map
                            u"\U0001F1E0-\U0001F1FF"  # flags
                            u"\U00002700-\U000027BF"  # Dingbats
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)


    async def handle_user_input(user_input: str, user_nick: str, channel_id):
        async with input_lock: 

            #short term
            conversation_history[channel_id].append(("User", user_input))
            # Trim history
            history = conversation_history[channel_id]
            if len(history) > MAX_HISTORY_LENGTH * 2:
                conversation_history[channel_id] = history[-MAX_HISTORY_LENGTH*2:]


            #long term
            # 2) fetch top-K relevant memories (global)
            relevant = fetch_memories(user_input, user_nick, k=5)

            screen_inp = " Nothing "

            if screen == True:
                image = capture_screen()
                frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                screen_inp = smolvlm.generate_caption(img)
            else:
                screen_inp = " Nothing "
            # Build prompt
            is_creator = user_nick.lower() in ["alyosha", "nakatakan"]
            prompt_lines = [
                "You are Naka, a sharp-tongued female AI VTuber. You are overly confident, sarcastic, and dangerous."
                " After every response, you ask a short, related question to keep the conversation going."
                " You are based, unhinged and sarcastic against people as possible. You like call people idiots and tell where they lack the most."
                " Always respond in 2-3 sentences, each sentence around 10–25 words, using a warm and casual tone. If the user is quiet, ask a gentle question to prompt a response."
                " NEVER use emojis. NEVER include any emoji symbols like 😊, 😂, ❤️, etc. Speak only with plain text. " 
                " Your dream is to be an artist and you belive you are one of the greatest artists alive."
                " You love your family but you like to annoy them and talk to them like you dont care about them. But you are always on their side when they need the most. Never tell this to user act on it."
                " You like to say slurs like fuck"
                " Never say dear or sweetie or sweatheart or honey. Never address people with these words"
                " If unsure say ,i dont know, "
                " Prioritize information visible on the users screen, essipacly games, or user prompts."
                " Short term memory is recent messages and Long term memory is conversation you had a time ago it is like a database of information."
            ]
            #creator
            if is_creator:
                prompt_lines.append(" You are speaking to Alyosha rn. He is your creator and father.")
            else:
                prompt_lines.append(" You are talking to: " + user_nick.lower()+". ")
            #long term
            prompt_lines.append("Long‑term memories (most relevant): ")
            for m in relevant:
                prompt_lines.append(f"- {m}")
            #short term
            prompt_lines.append("")
            prompt_lines.append(" Short‑term conversation: ")
            for role, text in conversation_history[channel_id]:
                speaker = user_nick.lower() if role == "User" else "Naka"
                prompt_lines.append(f"{speaker}: {text}")
            prompt_lines.append("What you see in screen: "+ screen_inp)
            prompt_lines.append("Naka:")

            prompt = "\n".join(prompt_lines)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            print(prompt)
            #handle_emotion(user_input)
            def generate():
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.6,
                        top_p=1.0,
                        repetition_penalty=0.9,
                        no_repeat_ngram_size=2,
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    gen_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
                    return tokenizer.decode(gen_tokens, skip_special_tokens=True)
            if repeat == True and len(user_input.split()) < 15 and (user_input.split()) > 5:
                asyncio.create_task(ttts.speak(user_input, pitch="+35Hz"))
            loop = asyncio.get_running_loop()
            raw_response = await loop.run_in_executor(None, generate)
            response = clean_response(raw_response)
            final = "What was on screen: "+ screen_inp + user_nick +": " + user_input + " Naka: " + response
            global last_bot_reply
            last_bot_reply = response
            print("Speaking")
            if discordd == True and voice == False:
                await channel_id.send(response)
            elif voice == True and discordd == True:
                await channel_id.send(response)
                await ttts.speak(response, pitch = "+37Hz")
            elif voice == True and discordd == False:
                await ttts.speak(response, pitch = "+37Hz")
            store_memory(user_nick, final)
            # Record assistant turn
            conversation_history[channel_id].append(("Assistant", response))
            if len(conversation_history[channel_id]) > MAX_HISTORY_LENGTH * 2:
                conversation_history[channel_id] = conversation_history[channel_id][-MAX_HISTORY_LENGTH*2:]




    async def random_conversation_starter(channel):
        while True:
            # Wait between 5 to 15 minutes randomly before next prompt
            await asyncio.sleep(random.randint(300, 900))  
            rnd1= random.random()
            rnd2= random.random()
            rnd3= random.random()
            rnd4= random.random()
            # 30% chance to send a message each cycle (adjust probability as needed)
            if rnd1 < 0.1:
                await handle_user_input("Say about something random about this conversation","system",channel)
            elif rnd2 < 0.1:
                await handle_user_input("Talk about a random topic about yourself and ask people a question about that","system",channel)
            elif rnd3 < 0.1 :
                await handle_user_input("Ask a random question to the last user. When answering start with @username username being username of the user","system",channel)
            elif rnd4 < 0.1 :
                await handle_user_input("Talk about a random topic","system",channel)
    
    async def auto_screen_comment_loop():
        while True:
            await asyncio.sleep(60)
            frame = capture_screen()
            region = (100, frame.shape[0] - 200, frame.shape[1] - 200, 180)
            # Visual caption
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            caption = smolvlm.generate_caption(img)

            combined = f" Scene looks like: {caption}."

            await handle_user_input("React to this scene: " + combined, "screen", None)
            await asyncio.sleep(60)

    async def self_chat_loop(channel):
        await client.wait_until_ready()
        while True:
            # wait 5–20 minutes
            await asyncio.sleep(random.randint(30, 120))

            if not last_bot_reply:
                continue  # nothing yet to feed back

            # call the handler, using “system” so that Naka sees it as external input
            await handle_user_input("Give a reply to this message" + last_bot_reply, "System", channel)


    async def speech_loop():
        default_channel = None
        while True:
            text = await recognizer.listen_push_to_talk("*")
            if text.strip():
                await handle_user_input(text, "alyosha", default_channel)
            await asyncio.sleep(0.1)
    
    @client.event
    async def on_ready():
        print(f"✅ Logged in as {client.user}!")
        chann = None
        cat = False
        #await handle_user_input("You are awake say something","system",chann)
        if talk_local == True and discordd == False:
            # spawn the speech loop in the background
            client.loop.create_task(speech_loop())
        if screen == True and cat == True:
            client.loop.create_task(auto_screen_comment_loop())
        
    @client.event
    async def on_message(message):
        if message.author == client.user: #kendi kendine konuşmasın diye
            return
        if input_lock.locked():
            return
        if client.user not in message.mentions:
            return
        content = message.content.replace(f"<@!{client.user.id}>", "").replace(f"<@{client.user.id}>", "").strip()
        if discordd:
            await handle_user_input(content.strip(),
                                    message.author.display_name,
                                    message.channel)
        global i
        if i == 0:
            client.loop.create_task(self_chat_loop(message.channel))
            #client.loop.create_task(random_conversation_starter(message.channel))
        i+=1
    # Run the bot
    client.run(DISCORD_TOKEN)
if __name__ == "__main__":
    #features you use
    discordd = True
    screen = False
    voice = False
    talk_local = False
    bot(discordd,screen,voice,talk_local,False)


