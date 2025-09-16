import os
import streamlit as st
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GroqHandler:
    def __init__(self):
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.api_key = os.getenv("GROOK_API_KEY")
        if not self.api_key:
            st.error("❌ GROOK_API_KEY not found in environment variables")

    def call_groq_api(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.1
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                return None
        except requests.exceptions.Timeout:
            st.warning("⏱️ Request timed out, please try again later...")
            return None
        except Exception as e:
            st.error(f"Error calling Groq API: {str(e)}")
            return None

    def get_response(self, prompt):
        response = self.call_groq_api(prompt)
        if response and len(response.strip()) > 5:
            st.success("✅ Response from Groq model")
            return response
        return ("I apologize, but the AI model is currently unavailable. "
                "Please try again in 1-2 minutes, or contact support@company.com for immediate help.")

@st.cache_resource
def load_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
        st.success("✅ Knowledge base loaded!")
        return vectorstore
    except Exception as e:
        st.error(f"❌ Error loading knowledge base: {str(e)}")
        return None

def main():
    st.title("🤗 Customer Support Bot - Debug Mode")
    st.write("Welcome to Support Customer Service")

    # Show system status
    with st.sidebar:
        st.header("🔍 System Debug")
        st.write("**Status Check:**")
        # Test internet connection
        try:
            response = requests.get("https://httpbin.org/get", timeout=5)
            if response.status_code == 200:
                st.success("✅ Internet connection OK")
            else:
                st.error("❌ Internet connection issues")
        except:
            st.error("❌ No internet connection")
        # Show API key status
        api_key = os.getenv('GROOK_API_KEY')
        if api_key:
            st.success(f"✅ API Key loaded: {api_key[:10]}...{api_key[-4:]}")
        else:
            st.error("❌ No API key found")
        
        # Test Groq API
        try:
            test_url = "https://api.groq.com/openai/v1/chat/completions"
            test_payload = {"model": "openai/gpt-oss-120b", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}
            test_headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            test_response = requests.post(test_url, headers=test_headers, json=test_payload, timeout=10)
            if test_response.status_code == 200:
                st.success("✅ Groq API reachable")
            else:
                st.warning(f"⚠️ Groq API returned: {test_response.status_code}")
                st.code(test_response.text)
        except Exception as e:
            st.error(f"❌ Groq API error: {str(e)}")

    # Initialize
    groq_handler = GroqHandler()
    vectorstore = load_vectorstore()

    if not vectorstore:
        st.error("❌ Please run 'python ingest.py' first!")
        st.stop()

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm in debug mode. Ask me a question and I'll show you exactly what happens!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question to test the system..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            # Show what's happening step by step
            st.write("🔍 **Step 1:** Searching knowledge base...")
            docs = vectorstore.similarity_search(prompt, k=2)
            if docs:
                st.success(f"✅ Found {len(docs)} relevant documents")
                with st.expander("📄 View Found Documents"):
                    for i, doc in enumerate(docs):
                        st.write(f"**Document {i+1}:** {doc.page_content[:200]}...")
            else:
                st.warning("⚠️ No relevant documents found")
            st.write("🔍 **Step 2:** Creating AI prompt...")
            context = "\n".join([doc.page_content for doc in docs])
            full_prompt = f"Answer this customer support question based on the context:\n\nContext: {context}\n\nQuestion: {prompt}\n\nHelpful Answer:"
            with st.expander("📝 View Full Prompt Sent to AI"):
                st.code(full_prompt)
            st.write("🔍 **Step 3:** Getting AI response...")
            response = groq_handler.get_response(full_prompt)
            st.write("🔍 **Step 4:** Final response:")
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
