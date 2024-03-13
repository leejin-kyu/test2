
import streamlit as st
import requests
import io
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import AnalyzeDocumentChain, RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

# 환경변수 설정
os.environ["OPENAI_API_KEY"] = "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"

# TXT 파일에서 텍스트를 불러오는 함수
@st.cache_data(hash_funcs={str: id})
def get_text_from_txt(txt_url):
    # 구글 드라이브 파일 ID 추출
    file_id = txt_url.split('/')[-2]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # 파일을 메모리에 직접 다운로드
    response = requests.get(download_url)
    text_all = io.StringIO(response.content.decode("utf-8")).read()
    return text_all

# TXT 파일에서 텍스트 추출 및 질문에 대한 답변 제공 함수
def answer_question_from_txt(question):
    # 고정된 TXT 파일 URL 설정
    txt_file_url = "https://drive.google.com/file/d/13SGsUt-QRYOoE8Wgu6YD9-TYHbhK6ZUe/view?usp=drive_link"
    st.write("텍스트 문서에서 정보를 검색 중입니다. 잠시만 기다려 주세요.")    
    text_all = get_text_from_txt(txt_file_url)
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=20)
    texts = text_splitter.split_text(text_all)
    docsearch = FAISS.from_texts(texts=texts, embedding=OpenAIEmbeddings())
    model = ChatOpenAI(model="gpt-4", temperature=0, max_tokens=1000)
    qa_chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=docsearch.as_retriever())
    
    answer = qa_chain.run(query=question)
    st.write("- 질문:", question)
    st.write("- 답변:", answer)

# 메인 화면 구성
st.title("농협 해외협력팀 파일럿 챗봇 ver1.0")

question = st.text_input("질문을 입력하세요.")

if st.button('답변 검색'):
    answer_question_from_txt(question)
