import sqlite3
import json

# Connect to your SQLite database
conn = sqlite3.connect(r"C:\Users\atoca\Desktop\Naka-chan\local model\Naka-Brain\Memories\memo.db")
cursor = conn.cursor()

# Query all rows from the memories table
cursor.execute("SELECT text FROM memories;")
rows = cursor.fetchall()

output_lines = []

# Extract and structure the conversation pairs
for row in rows:
    text = row[0]
    if "yax:" in text and "Naka:" in text:
        try:
            user_part = text.split("yax:")[-1].split("Naka:")[0].strip()
            assistant_part = text.split("Naka:")[-1].strip()
            
            # Format into the desired JSON structure
            message = {
                "messages": [
                    {"role": "user", "content": user_part},
                    {"role": "assistant", "content": assistant_part}
                ]
            }

            # Dump each line as a separate JSON object
            output_lines.append(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            continue  # Skip malformed rows

# Save to a file
with open(r"C:\Users\atoca\Desktop\Naka-chan\local model\Naka-Brain\Memories\formatted_conversations.jsonl", "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"Extracted {len(output_lines)} message pairs to 'formatted_conversations.jsonl'")
