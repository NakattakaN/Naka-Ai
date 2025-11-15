from sentence_transformers import SentenceTransformer

# First-time download
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save(r"C:\Users\atoca\Desktop\Naka-chan\local model\Naka-Brain\Memories/hipocam")


model = SentenceTransformer(r"C:\Users\atoca\Desktop\Naka-chan\local model\Naka-Brain\Memories/hipocam")
print(model.encode(["test"]))
