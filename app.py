# app.py
import streamlit as st
import pdfplumber
from rag_pipeline import create_rag_pipeline
from ml_analyzer import DocumentAnalyzer
# --- ADD THIS IMPORT ---
from langchain.text_splitter import RecursiveCharacterTextSplitter 

st.set_page_config(page_title="IntelliDoc - RAG + ML Assistant", page_icon="📘")
st.title("📘 IntelliDoc - RAG + ML Smart Document Assistant")

# Upload PDFs
uploaded_files = st.file_uploader(
    "📂 Upload your PDF documents", 
    accept_multiple_files=True, 
    type=["pdf"]
)

if uploaded_files:
    st.info("⏳ Extracting text from uploaded PDFs...")
    all_text = ""

    # Extract text from PDFs
    for pdf in uploaded_files:
        try:
            with pdfplumber.open(pdf) as pdf_doc:
                for page in pdf_doc.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")

    if not all_text.strip():
        st.warning("⚠️ No readable text found in the uploaded PDFs.")
    else:
        st.success("✅ Documents loaded and processed successfully!")

        # --- FIX 1: SPLIT TEXT INTO CHUNKS ---
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Split into 1000-character chunks
            chunk_overlap=150 # Overlap chunks by 150 characters
        )
        doc_chunks = text_splitter.split_text(all_text)

        if not doc_chunks:
            st.warning("⚠️ Could not extract any text chunks for RAG.")
        else:
            # --- END FIX 1 ---
            
            # Create RAG pipeline
            st.info(f"🔗 Creating RAG pipeline with {len(doc_chunks)} text chunks...")
            try:
                # --- FIX 2: PASS THE CHUNKS TO THE PIPELINE ---
                qa_chain = create_rag_pipeline(doc_chunks)
                # --- END FIX 2 ---

                analyzer = DocumentAnalyzer()
                
                # User query
                query = st.text_input("💬 Ask a question about your documents:")

                if query:
                    with st.spinner("Thinking... 🧠"):
                        try:
                            result = qa_chain.invoke({"query": query})
                            answer = result.get("result", str(result))
                        except Exception as e:
                            answer = f"Error while generating answer: {e}"

                        # Optional: ML-based analysis
                        try:
                            sentiment = analyzer.predict(query)
                        except FileNotFoundError:
                            sentiment = "N/A (Model not trained yet)"
                        except Exception as e:
                            sentiment = f"N/A (Error: {e})"

                        # Display results
                        st.markdown("### 🧾 Answer:")
                        st.markdown(f"> {answer}")

                        st.markdown("### 🤖 ML Analysis:")
                        st.markdown(f"**Predicted Sentiment / Tone:** {sentiment}")
            
            except Exception as e:
                st.error(f"Error creating RAG pipeline: {e}")
else:
    st.info("📄 Please upload one or more PDF files to begin.")