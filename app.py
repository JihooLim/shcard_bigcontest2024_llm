import os
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import faiss

import streamlit as st
#가나라
# 경로 설정
data_path = './data'
module_path = './modules'

# Gemini 설정
import google.generativeai as genai

GOOGLE_API_KEY = st.secrets["API_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)

# Gemini 모델 선택
model = genai.GenerativeModel("gemini-1.5-flash")


# CSV 파일 로드
## 자체 전처리를 거친 데이터 파일 활용
csv_file_path = "JEJU_DATA.csv"
df = pd.read_csv(os.path.join(data_path, csv_file_path),encoding = 'cp949')

# 최신연월 데이터만 가져옴
df = df[df['기준연월'] == df['기준연월'].max()].reset_index(drop=True)


# Streamlit App UI

st.set_page_config(page_title="🍊제주도 맛집 추천")



# Replicate Credentials
with st.sidebar:
    st.title("**🍊제주도 맛집 추천**")

    st.write("")
    st.markdown("""
        <style>
        .sidebar-text {
        color: #B271FF;
        font-size: 20px;
        font-weight: bold;
        }
     </style>
     """, unsafe_allow_html=True)

    st.sidebar.markdown('<p class="sidebar-text">시간대가 어떻게 되시나요??</p>', unsafe_allow_html=True)

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

    time = st.sidebar.selectbox("", ["상관 없음","아침", "점심", "오후", "저녁", "밤"], key="time")

    st.write("")

    st.sidebar.markdown('<p class="sidebar-text">희망 가격대는 어떻게 되시나요??</p>', unsafe_allow_html=True)

    # radio 레이블 공백 제거
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

    price = st.sidebar.selectbox("", ['상관 없음','최고가', '고가', '평균 가격대', '중저가', '저가'], key="price")
    

    st.markdown(
        """
         <style>
         [data-testid="stSidebar"] {
         background-color: #ff9900;
         }
        </style>
        """, unsafe_allow_html=True)
    st.write("")

st.title("어서 와용!👋")
st.subheader("인기있는 :orange[제주 맛집]🧑 후회는 없을걸?!")

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
    st.session_state.messages = [{"role": "assistant", "content": "어떤 식당 찾고 있나요?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "어떤 식당 찾고 있나요?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# RAG

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face의 사전 학습된 임베딩 모델과 토크나이저 로드
model_name = "jhgan/ko-sroberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to(device)

print(f'Device is {device}.')


# FAISS 인덱스 로드 함수
def load_faiss_index(index_path=os.path.join(module_path, 'faiss_index.index')):
    """
    FAISS 인덱스를 파일에서 로드합니다.

    Parameters:
    index_path (str): 인덱스 파일 경로.

    Returns:
    faiss.Index: 로드된 FAISS 인덱스 객체.
    """
    if os.path.exists(index_path):
        # 인덱스 파일에서 로드
        index = faiss.read_index(index_path)
        print(f"FAISS 인덱스가 {index_path}에서 로드되었습니다.")
        return index
    else:
        raise FileNotFoundError(f"{index_path} 파일이 존재하지 않습니다.")

# 텍스트 임베딩
def embed_text(text):
    # 토크나이저의 출력도 GPU로 이동
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        # 모델의 출력을 GPU에서 연산하고, 필요한 부분을 가져옴
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()  # 결과를 CPU로 이동하고 numpy 배열로 변환

# 임베딩 로드
embeddings = np.load(os.path.join(module_path, 'embeddings_array_file.npy'))

def generate_response_with_faiss(question, df, embeddings, model, embed_text, time, local_choice, index_path=os.path.join(module_path, 'faiss_index.index'), max_count=10, k=10, print_prompt=True):
    filtered_df = df

    # FAISS 인덱스를 파일에서 로드
    index = load_faiss_index(index_path)

    # 검색 쿼리 임베딩 생성
    query_embedding = embed_text(question).reshape(1, -1)

    # 가장 유사한 텍스트 검색 (3배수)
    distances, indices = index.search(query_embedding, k*3)

    # FAISS로 검색된 상위 k개의 데이터프레임 추출
    filtered_df = filtered_df.iloc[indices[0, :]].copy().reset_index(drop=True)


    # 웹페이지의 사이드바에서 선택하는 영업시간, 현지인 맛집 조건 구현

    # 영업시간 옵션
    # 필터링 조건으로 활용

    # 영업시간 조건을 만족하는 가게들만 필터링
    if time == '상관 없음':
        filtered_df = filtered_df
    elif time == '아침':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(5, 12)))].reset_index(drop=True)
    elif time == '점심':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(12, 14)))].reset_index(drop=True)
    elif time == '오후':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(14, 18)))].reset_index(drop=True)
    elif time == '저녁':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in range(18, 23)))].reset_index(drop=True)
    elif time == '밤':
        filtered_df = filtered_df[filtered_df['영업시간'].apply(lambda x: isinstance(eval(x), list) and any(hour in eval(x) for hour in [23, 24, 1, 2, 3, 4]))].reset_index(drop=True)

    # 필터링 후 가게가 없으면 메시지를 반환
    if filtered_df.empty:
        return f"현재 선택하신 시간대({time})에는 영업하는 가게가 없습니다."

    filtered_df = filtered_df.reset_index(drop=True)

    # 희망 가격대 조건을 만족하는 가게들만 필터링
    if price == '상관 없음':
        filtered_df = filtered_df
    elif price == '최고가':
        filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith('6')].reset_index(drop=True)
    elif price == '고가':
        filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith('5'or'4')].reset_index(drop=True)
    elif price == '평균 가격대':
        filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith('3')].reset_index(drop=True)
    elif price == '중저가':
        filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith('2')].reset_index(drop=True)
    elif price == '저가':
        filtered_df = filtered_df[filtered_df['건당평균이용금액구간'].str.startswith('1')].reset_index(drop=True)
 
    # 필터링 후 가게가 없으면 반환
    if filtered_df.empty:
        return f"현재 선택하신 시간대({time})에는 영업하는 가게가 없습니다."

    filtered_df = filtered_df.reset_index(drop=True).head(k)


    # 선택된 결과가 없으면 처리
    if filtered_df.empty:
        return "질문과 일치하는 가게가 없습니다."


    # 참고할 정보와 프롬프트 구성
    reference_info = ""
    for idx, row in filtered_df.iterrows():
        reference_info += f"{row['text']}\n"

    # 응답을 받아오기 위한 프롬프트 생성
    prompt = f"질문: {question} 특히 {price}을 선호해\n참고할 정보:\n{reference_info}\n응답:"

    if print_prompt:
        print('-----------------------------'*3)
        print(prompt)
        print('-----------------------------'*3)

    # 응답 생성
    response = model.generate_content(prompt)

    return response


# User-provided prompt
if prompt := st.chat_input(): # (disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # response = generate_llama2_response(prompt)
            response = generate_response_with_faiss(prompt, df, embeddings, model, embed_text, time, price)
            placeholder = st.empty()
            full_response = ''

            # 만약 response가 GenerateContentResponse 객체라면, 문자열로 변환하여 사용합니다.
            if isinstance(response, str):
                full_response = response
            else:
                full_response = response.text  # response 객체에서 텍스트 부분 추출

            # for item in response:
            #     full_response += item
            #     placeholder.markdown(full_response)

            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
