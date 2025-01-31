import streamlit as st
import json
import os
import time
import sys
from dotenv import load_dotenv
import requests
import yt_dlp
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
from sentence_transformers import SentenceTransformer
import faiss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure required directories exist
os.makedirs('temp', exist_ok=True)
os.makedirs('docs', exist_ok=True)

load_dotenv()
api_token = os.getenv('ASSEMBLY_AI_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

base_url = "https://api.assemblyai.com/v2"

headers = {
    "authorization": api_token,
    "content-type": "application/json"
}

# yt-dlp function for YouTube video
def save_audio(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'temp/%(title)s.%(ext)s',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_filename = ydl.prepare_filename(info).replace('.webm', '.mp3')
        
        logger.info(f"Successfully downloaded audio: {audio_filename}")
        return Path(audio_filename).name
    except Exception as e:
        logger.error(f"Error downloading audio: {str(e)}")
        st.error(f"Error downloading audio: {str(e)}")
        return None

def assemblyai_stt(audio_filename):
    try:
        audio_path = os.path.join('temp', audio_filename)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        with open(audio_path, "rb") as f:
            response = requests.post(base_url + "/upload",
                                    headers=headers,
                                    data=f)
        response.raise_for_status()

        upload_url = response.json()["upload_url"]
        data = {
            "audio_url": upload_url
        }
        url = base_url + "/transcript"
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        transcript_id = response.json()['id']
        polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

        while True:
            transcription_result = requests.get(polling_endpoint, headers=headers).json()

            if transcription_result['status'] == 'completed':
                break
            elif transcription_result['status'] == 'error':
                raise RuntimeError(f"Transcription failed: {transcription_result['error']}")
            else:
                time.sleep(3)
                
        transcription_text = transcription_result['text']
        word_timestamps = transcription_result['words']
        
        with open('docs/transcription.txt', 'w', encoding='utf-8') as file:
            file.write(transcription_text)
        with open('docs/word_timestamps.json', 'w', encoding='utf-8') as file:
            json.dump(word_timestamps, file)
        
        logger.info("Successfully transcribed audio with word-level timestamps")
        return transcription_text, word_timestamps
    except Exception as e:
        logger.error(f"Error in speech-to-text conversion: {str(e)}")
        st.error(f"Error in speech-to-text conversion: {str(e)}")
        return None, None

@st.cache_resource
def setup_qa_chain(transcription_path='docs/transcription.txt'):
    try:
        if not os.path.exists(transcription_path):
            st.error("Transcription file not found!")
            return None, None

        loader = TextLoader(transcription_path)
        documents = loader.load()

        # Split text into chunks for indexing
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # Use SentenceTransformers for embeddings
        model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        embeddings = model.encode([doc.page_content for doc in texts])

        # Create a FAISS index using CPU
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        # Create docstore and index_to_docstore_id (mapping documents to their IDs)
        docstore = {i: texts[i] for i in range(len(texts))}  # Map index to document
        index_to_docstore_id = {i: str(i) for i in range(len(texts))}  # Simple index to ID mapping

        # Create FAISS vector store
        vectorstore = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=model.encode)
        retriever = vectorstore.as_retriever()

        chat = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Read word timestamps
        word_timestamps_path = 'docs/word_timestamps.json'
        if os.path.exists(word_timestamps_path):
            with open(word_timestamps_path, 'r') as file:
                word_timestamps = json.load(file)
        else:
            word_timestamps = []

        return qa_chain, word_timestamps
    except Exception as e:
        logger.error(f"Error setting up QA chain: {str(e)}")
        st.error(f"Error setting up QA chain: {str(e)}")
        return None, None

def find_relevant_timestamps(answer, word_timestamps):
    relevant_timestamps = []
    answer_words = answer.lower().split()
    for word_info in word_timestamps:
        if word_info['text'].lower() in answer_words:
            relevant_timestamps.append(word_info['start'])
    return relevant_timestamps

def generate_summary(transcription):
    client = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile")
    summary_prompt = PromptTemplate(
        input_variables=["transcription"],
        template="Summarize the following transcription in 3-5 sentences:\n\n{transcription}"
    )
    summary_chain = LLMChain(llm=client, prompt=summary_prompt)
    summary = summary_chain.run(transcription)
    return summary

# Streamlit UI
st.set_page_config(layout="wide", page_title="ChatAudio", page_icon="🔊")

st.title("Chat with Your Audio using Groq LLM")

# Default values
transcription = None
word_timestamps = None
qa_chain = None

input_source = st.text_input("Enter the YouTube video URL")

if input_source:
    col1, col2 = st.columns(2)

    with col1:
        st.info("Your uploaded video")
        st.video(input_source)
        audio_filename = save_audio(input_source)
        
        if audio_filename:
            transcription, word_timestamps = assemblyai_stt(audio_filename)
            
            if transcription:
                st.info("Transcription completed. You can now ask questions.")
                st.text_area("Transcription", transcription, height=300)
                
                # Initialize QA chain
                qa_chain, word_timestamps = setup_qa_chain()

                if st.button("Generate Summary"):
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(transcription)
                        st.subheader("Summary")
                        st.write(summary)

    with col2:
        st.info("Chat Below")
        query = st.text_input("Ask your question here...")
        
        if query and qa_chain is not None:
            with st.spinner("Generating answer..."):
                try:
                    # Query the qa_chain correctly
                    result = qa_chain({"query": query})  # Pass the query as a dictionary
                    answer = result.get('result', 'No answer found')  # Safely get the result
                    
                    st.success(answer)
                    
                    relevant_timestamps = find_relevant_timestamps(answer, word_timestamps)
                    if relevant_timestamps:
                        st.subheader("Relevant Timestamps")
                        for timestamp in relevant_timestamps[:5]:  # Limit to top 5 timestamps
                            st.write(f"{timestamp // 60}:{timestamp % 60:02d}")
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

# Cleanup temporary files when the script exits
def cleanup_temp_files():
    for file in os.listdir('temp'):
        os.remove(os.path.join('temp', file))
