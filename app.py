import os
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import faiss

import streamlit as st

# 경로 설정
data_path = './data'
module_path = './modules'

# Gemini 설정
import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyBx1J1pS9k7bNA7R-5fkgAK8K7xQxd7Fes"

genai.configure(api_key=GOOGLE_API_KEY)

# Gemini 모델 선택
model = genai.GenerativeModel("gemini-1.5-flash")


# CSV 파일 로드
## 자체 전처리를 거친 데이터 파일 활용
csv_file_path = "JEJU_MCT_DATA_modified.csv"
df = pd.read_csv(os.path.join(data_path, csv_file_path))

# 최신연월 데이터만 가져옴
df = df[df['기준연월'] == df['기준연월'].max()].reset_index(drop=True)


# Streamlit App UI

st.set_page_config(page_title="🍊제주도가 그리 좋아!")

# Replicate Credentials
with st.sidebar:
    st.title("🍊제주도가 그리 좋아!")

    st.write("")

    st.subheader("시간대가 어떻게 돼??")

    # selectbox 레이블 공백 제거
    st.markdown(
        """
        <style>
        .stSelectbox label {  /* This targets the label element for selectbox */
            display: none;  /* Hides the label element */
        }
        .stSelectbox div[role='combobox'] {
            margin-top: -20px; /* Adjusts the margin if needed */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    time = st.sidebar.selectbox("", ["아침", "점심", "오후", "저녁", "밤"], key="time")

    st.write("")

    st.subheader("현지인 맛집? 관광객 맛집?")

    # radio 레이블 공백 제거
    st.markdown(
        """
        <style>
        .stRadio > label {
            display: none;
        }
        .stRadio > div {
            margin-top: -20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    local_choice = st.radio(
        '',
        ('제주도민 맛집', '관광객 맛집')
    )

    st.write("")

st.title("어서와용!👋")
st.subheader("인기있는 제주 맛집🧑‍🍳 후회는 없을걸?!")

st.write("")

st.write("#흑돼지 #제철 생선회 #해물라면 #스테이크 #한식 #중식 #양식 #일식 #흑백요리사..🤤")

st.write("")

image_path = "https://pimg.mk.co.kr/news/cms/202409/22/news-p.v1.20240922.a626061476c54127bbe4beb0aa12d050_P1.png"
image_html = f"""
<div style="display: flex; justify-content: center;">
    <img src="{image_path}" alt="centered image" width="50%">
</div>
"""
st.markdown(image_html, unsafe_allow_html=True)

st.write("")

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "어드런 식당 찾으시쿠과?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "어드런 식당 찾으시쿠과?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# RAG

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# FAISS 설정
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to("cpu")

# FAISS 인덱스 로드 함수
def load_faiss_index(index_path=os.path.join(module_path, 'faiss_index.index')):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        return index
    else:
        raise FileNotFoundError(f"{index_path} 파일이 존재하지 않습니다.")

# 텍스트 임베딩 함수
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to("cpu")
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

# FAISS + Gemini 응답 생성 함수
def generate_response_with_priority(question, df, embeddings, model, embed_text, time, local_choice, index_path=os.path.join(module_path, 'faiss_index.index'), k=3):
    # 1. FAISS 인덱스를 통해 질문과 유사한 데이터를 검색
    try:
        index = load_faiss_index(index_path)
    except:
        st.error(f"FAISS 인덱스를 찾을 수 없습니다: {e}")
        return None
    
    query_embedding = embed_text(question).reshape(1, -1)
    distances, indices = index.search(query_embedding, k*3)
    
    # 검색된 상위 k개의 데이터 추출
    filtered_df = df.iloc[indices[0, :]].copy().reset_index(drop=True)

    # 영업시간 옵션
    if time == '아침':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(5, 12)))].reset_index(drop=True)
    elif time == '점심':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(12, 14)))].reset_index(drop=True)
    elif time == '오후':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(14, 18)))].reset_index(drop=True)
    elif time == '저녁':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(18, 23)))].reset_index(drop=True)
    elif time == '밤':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in [23, 24, 1, 2, 3, 4]))].reset_index(drop=True)

       # 2. 데이터가 있을 경우, 이를 기반으로 답변 생성
    if not filtered_df.empty:
        response_text = "\n".join([row['text'] for _, row in filtered_df.iterrows()])
        return f"다음과 같은 추천이 있습니다:\n{response_text}"

    # 3. 데이터가 없을 경우, Gemini에게 질문을 넘겨서 대답을 생성
    else:
        # Gemini 모델에 질문을 전달하고 답변 생성
        prompt = f"질문: {question} 특히 {local_choice}을 선호해"
        response = model.generate_content(prompt)
        except:
            st.error(f"Gemini에서 답변을 생성하는 중 오류가 발생했습니다: {e}")
            return None
        return response


# 임베딩 로드 확인
embeddings_path = os.path.join(module_path, 'embeddings_array_file.npy')
if os.path.exists(embeddings_path):
    embeddings = np.load(embeddings_path)
else:
    st.error(f"{embeddings_path} 파일을 찾을 수 없습니다.")
    embeddings = None

# 유저 질문 입력 처리
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response with priority (FAISS -> Gemini)
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성 중입니다..."):
                response = generate_response_with_priority(prompt, df, embeddings, model, embed_text, time, local_choice)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

   
