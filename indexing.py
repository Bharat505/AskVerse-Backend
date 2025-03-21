import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document  # Fixing Chroma issue
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader, UnstructuredHTMLLoader

###############################################################################
# Configuration
###############################################################################

# ✅ Define folders where documents are stored
FOLDER_PATHS = ["kafka", "react", "spark"]

# ✅ Load API Key from Environment Variable
openai_api_key = os.getenv("OPENAI_API_KEY")
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)

# ✅ Define ChromaDB storage path
CHROMA_DB_PATH = "./chroma_db"

###############################################################################
# Step 1: Reset and Initialize ChromaDB
###############################################################################

def reset_chromadb():
    """ Clears and resets ChromaDB. """
    if os.path.exists(CHROMA_DB_PATH):
        print("🧹 Clearing existing ChromaDB...")
        for file in Path(CHROMA_DB_PATH).glob("*"):
            try:
                file.unlink()  # Delete files inside chroma_db
            except Exception as e:
                print(f"⚠️ Error deleting {file}: {e}")
        print("✅ ChromaDB reset complete!")

# ✅ Reset and initialize ChromaDB
reset_chromadb()
vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)

###############################################################################
# Step 2: Load Documents in Parallel
###############################################################################

def load_documents(folder):
    """ Loads Markdown and HTML documents from a given folder. """
    all_docs = []
    for file_path in Path(folder).rglob("*"):
        if file_path.suffix == ".md":
            loader = UnstructuredMarkdownLoader(str(file_path))
        elif file_path.suffix == ".html":
            loader = UnstructuredHTMLLoader(str(file_path))
        else:
            continue  # Skip unsupported files

        try:
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(file_path)  # Store file source separately
            all_docs.extend(docs)
            print(f"✅ Loaded: {file_path}")
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")

    return all_docs

# ✅ Load all documents from folders using parallel processing
def load_all_documents():
    """ Loads all documents from FOLDER_PATHS in parallel. """
    all_docs = []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_documents, FOLDER_PATHS))
        for res in results:
            all_docs.extend(res)
    print(f"\n📄 Successfully loaded {len(all_docs)} documents (Markdown + HTML)!")
    return all_docs

all_docs = load_all_documents()

###############################################################################
# Step 3: Semantic Chunking
###############################################################################

class SemanticChunker:
    """
    Custom chunker that:
    - Preserves Markdown headers (`#` tags).
    - Keeps code blocks intact.
    - Uses sentence-aware chunking for coherence.
    """

    def __init__(self, chunk_size=5000, min_chunk_size=2500, overlap_ratio=0.2):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = int(chunk_size * overlap_ratio)

    def split_text(self, text):
        """
        Splits text using semantic awareness:
        - **Markdown Headers:** Sections are split at `#`, `##`, `###`.
        - **Code Blocks:** Keeps fenced code blocks (` ``` `).
        - **Sentence-aware splitting:** Ensures logical sentence breaks.
        """

        # ✅ Step 1: **First split by Markdown headers (`# Section Title`)**
        sections = re.split(r'(#+ .+)', text)  # Keep headers with their sections
        final_chunks = []
        current_chunk = []

        for section in sections:
            # ✅ Step 2: **Detect Code Blocks and Keep Them Together**
            if "```" in section:
                if current_chunk:
                    final_chunks.append(" ".join(current_chunk))
                    current_chunk = []
                final_chunks.append(section.strip())  # Store entire code block
                continue

            # ✅ Step 3: **Apply Sentence Splitting Inside Each Section**
            sentences = re.split(r'(?<=[.!?])\s+', section)  # Sentence-aware split
            for sentence in sentences:
                if sum(len(s) for s in current_chunk) + len(sentence) < self.chunk_size:
                    current_chunk.append(sentence)
                else:
                    if sum(len(s) for s in current_chunk) >= self.min_chunk_size:
                        final_chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]

        if current_chunk:
            final_chunks.append(" ".join(current_chunk))  # Add remaining chunk

        return final_chunks

###############################################################################
# Step 4: Apply Chunking and Store in ChromaDB
###############################################################################

def chunk_documents(docs):
    """ Chunks the documents using semantic chunking. """
    all_chunks = []
    chunker = SemanticChunker(chunk_size=5000, min_chunk_size=2500, overlap_ratio=0.2)

    for doc in docs:
        file_name = doc.metadata.get("source", "Unknown Source")
        chunks = chunker.split_text(doc.page_content)

        for chunk in chunks:
            doc_chunk = Document(
                page_content=chunk,
                metadata={"file_name": file_name, "source": file_name}
            )
            all_chunks.append(doc_chunk)

    return all_chunks

# ✅ Apply Semantic Chunking to Documents
split_docs = chunk_documents(all_docs)

# ✅ Store in ChromaDB
def index_documents():
    """ Indexes documents into ChromaDB. """
    if split_docs:
        vector_db.add_documents(split_docs)
        vector_db.persist()
        print("✅ Documents indexed with **Semantic Chunking**!")
        print(f"📌 Total documents indexed: {len(vector_db.get()['documents'])}")
    else:
        print("⚠️ No documents to index in ChromaDB!")

index_documents()

###############################################################################
# Run as Script
###############################################################################
if __name__ == "__main__":
    print("🔄 Indexing process completed successfully!")
