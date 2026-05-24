from pdf2image import convert_from_path
from image_embedder import ImageEmbedder
import os

# Path to a sample PDF file
pdf_path = os.path.join("saved_docs", "1706.03762_1.pdf")  # Change the filename as needed

try:
    # Convert first page of PDF to image
    images = convert_from_path(pdf_path, first_page=1, last_page=1)
    if not images:
        print("No images extracted from PDF.")
        exit(1)
    img = images[0]
except Exception as e:
    print(f"Failed to convert PDF to image: {e}")
    exit(1)

embedder = ImageEmbedder()
embedding = embedder.embed_image(img)

print("Embedding shape:", embedding.shape)
print("Embedding (first 10 values):", embedding[:10])
