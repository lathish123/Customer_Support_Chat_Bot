import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_sample_documents():
    """Create sample customer support documents"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Sample FAQ
    faq_content = """
Q: How do I reset my password?
A: Go to the login page, click "Forgot Password", enter your email, and follow the instructions sent to you.
Q: How long does shipping take?
A: Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.
Q: How do I track my order?
A: Log into your account and go to "My Orders" to see tracking information.
Q: What is your return policy?
A: We accept returns within 30 days of purchase. Items must be unused and in original packaging.
Q: How do I contact support?
A: Email us at support@company.com or call 1-800-SUPPORT. We're available 24/7.
    """

    # Sample policies
    policies_content = """
Shipping Policy:
- Free shipping on orders over $50
- International shipping available
- Orders ship Monday-Friday
Return Policy:
- 30-day return window
- Items must be unused
- Free return shipping for defective items
- Refunds processed in 5-7 business days
Privacy Policy:
- We protect your personal information
- Data is encrypted and secure
- We never sell your data to third parties
    """

    # Write files
    with open(data_dir / "faq.txt", "w") as f:
        f.write(faq_content)
    with open(data_dir / "policies.txt", "w") as f:
        f.write(policies_content)
    print("✅ Created sample documents in 'data' folder")

def main():
    print("🚀 Setting up your knowledge base...")
    # Create sample documents
    create_sample_documents()
    # Load documents
    docs = []
    data_dir = Path("data")
    for file_path in data_dir.glob("*.txt"):
        loader = TextLoader(str(file_path))
        docs.extend(loader.load())
    print(f"📚 Loaded {len(docs)} documents")
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"✂️ Split into {len(chunks)} chunks")
    # Create embeddings and save
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")
    print("✅ Knowledge base created successfully!")
    print("▶️ Now run: streamlit run app.py")

if __name__ == "__main__":
    main()
