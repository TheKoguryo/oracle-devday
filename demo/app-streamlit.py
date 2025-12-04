import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

import tempfile
import os

from oci_genai_utils import list_regions
from oci_genai_utils import list_chat_models
from oci_genai_utils import chat
from oci_genai_utils import chat_with_rag
from oci_genai_utils import change_region
from oci_genai_utils import load_to_vector_store

st.set_page_config(layout="wide")

hide_streamlit_style = """
    <style>
        [data-testid="stToolbar"] {visibility: hidden !important;}
        footer {visibility: hidden !important;}
    </style>
    """
#st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.markdown(
    """
<style>
    .st-emotion-cache-1c7y2kd {
        flex-direction: row-reverse;
        text-align: right;
        background-color: white;
    }

    div[data-testid="stChatMessageAvatarUser"] {
        visibility: hidden;
        width: 0;
    }
    div[data-testid="chatAvatarIcon-user"] {
        visibility: hidden;
        width: 0;
    }

    div[data-testid="stSidebarHeader"] {
        height: 2rem;
    }   

    .stVerticalBlock {
        justify-content: center;
    }
      
    div[data-testid="stMarkdown"] div[data-testid="stMarkdownContainer"] p {
        display: flex;
        justify-content: center;
        align-items: center;
    }    

    button[data-testid="stBaseButton-secondary"] {
        min-height: 0rem;
        width: 100%;
    }

    div[data-testid="stElementContainer"] {
        width: 100%;
    }

    div[data-testid="stMainBlockContainer"] div[data-testid="stMarkdownContainer"] p {
        justify-content: left;
    }

    div[data-testid="stMainBlockContainer"] div[aria-label="Chat message from user"] div[data-testid="stMarkdownContainer"] p {
        justify-content: right;
    }

    div[data-testid="stChatMessageContent"] div[aria-label="Chat message from user"] div[data-testid="stMarkdownContainer"] p {
        justify-content: right;
    }

    div[aria-label="Chat message from assistant"] div[data-testid="stMarkdown"] div[data-testid="stMarkdownContainer"] p {
        display: flex;
        justify-content: left;
        align-items: left;
    }

    div[aria-label="Chat message from user"] div[data-testid="stMarkdown"] div[data-testid="stMarkdownContainer"] p {
        display: flex;
        justify-content: right;
        align-items: right;
    }

    div[data-testid="stMainBlockContainer"] div.st-emotion-cache-1fee4w7 {
        margin-left: auto;
        margin-right: 0;
        width: max-content;
        padding: 0.5rem;
        border-radius: 1.25rem;
    }

    div[data-testid="stChatMessage"] div[aria-label="Chat message from user"] div.st-emotion-cache-uzeiqp p {
        margin-left: auto;
        margin-right: 0;
        width: max-content;
        padding: 0.25rem;
        border-radius: 1rem;
        background-color: beige;
        padding-left: 1rem;
        padding-right: 1rem;
    }         
</style>
""",
    unsafe_allow_html=True,
)

def when_region_name_changes():
    print(st.session_state['region_name'])
    change_region(st.session_state['region_name'])

def when_model_name_changes():
    print(st.session_state['model_name'])
    st.session_state.messages.append({"role": "assistant", "content": st.session_state['model_name'] + " 모델로 변경"})

with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

left, center, right = st.columns([1, 2, 1])
with center:
    authenticator.login(fields={'Form name':'OCI Generative AI Chat', 'Login':'로그인'})

    if st.session_state.get('authentication_status') is None:
        st.warning('Please enter your username and password')
    elif st.session_state.get('authentication_status') is False:
        st.error('Username/password is incorrect')

if st.session_state.get('authentication_status'):   
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        st.markdown(f"**{st.session_state.get('name', '')}**&nbsp;님")
    with col2:
        authenticator.logout('로그아웃')

    tab1, tab2 = st.sidebar.tabs(['검색', '인덱싱'])

    with tab1:
        st.header('검색')
        
        region_names, regions = list_regions()
        region_name_selected = st.selectbox("리전", region_names, key='region_name', on_change=when_region_name_changes)
        region_selected_index = region_names.index(region_name_selected)

        model_names, models = list_chat_models(region_name_selected)
        model_name = st.selectbox("모델", model_names, key='model_name', on_change=when_model_name_changes)
        model_selected_index = model_names.index(model_name)
        model_selected = models[model_selected_index]

        rag_option = st.radio(
            "RAG 사용 여부 선택",
            ["Without RAG", "With RAG"],
            captions=[
                "일반 질문",
                "수집된 자료에서 검색",
            ],
            horizontal=True,
        )

        if rag_option == "With RAG":
            search_category_option = st.radio(
                "검색 대상",
                ["ALL", "내부", "공개", "기타"],
                captions=[
                ],
                horizontal=True,
            )

        st.markdown("---")

        st.subheader("샘플 질문 예시")

        sample_questions = [
            """한국이 구축 예정인 AI 데이터 센터 규모, 예산은
어떻게 되나요?""",
            """글로벌 소프트웨어 기업의 오픈 소스 기여는 어떻게 되나요?
오라클의 오픈소스 기여순위는 얼마나 되나요?""",
            "삼성전자가 자체 개발한 AI 의 이름은?",
        ]

        #for q in sample_questions:
            #if st.button(q):
                #st.session_state.user_input = q
            #st.code(q, language=None)

        with st.container():
            for q in sample_questions:
                st.code(q, language="python")           
            

    with tab2:
        st.header('인덱싱 / 추가 문서 업로드')

        category_option = st.radio(
            "문서 분류",
            ["내부자료", "공개자료", "기타"],
            captions=[
            ],
            horizontal=True,
        )

        uploaded_file = st.file_uploader("PDF 업로드", type=["pdf"], key="uploaded_file")

        # 이전 업로드 상태 기억하기
        if "last_uploaded_filename" not in st.session_state:
            st.session_state.last_uploaded_filename = None    

        if uploaded_file is not None:
            if uploaded_file.name != st.session_state.last_uploaded_filename:
                st.session_state.last_uploaded_filename = uploaded_file.name
                print(uploaded_file)

            if st.button("인덱싱 시작하기", use_container_width=True):
                # 원본 파일명 가져오기
                original_filename = uploaded_file.name

                print(original_filename)

                # 임시 파일 생성
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                category = "unclassified"

                if category_option == "내부자료":
                    category = "internal"
                elif category_option == "공개자료":
                    category = "public"               

                result = load_to_vector_store(temp_file_path, original_filename, category)

                st.success("문서 추가 및 인덱싱 완료되었습니다!")

    # Title displayed on the streamlit web app
    #st.title(f""":green[OCI Generative AI Chat]""")

    # configuring values for session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        #open("chat_history.txt", "w").close()

    # writing the message that is stored in session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # adding some special effects from the UI perspective
    #st.balloons()

    # evaluating st.chat_input and determining if a question has been input
    if question:= st.chat_input("메시지 입력", key="user_input"):

        # with the user icon, write the question to the front end
        with st.chat_message("user"):
            st.markdown(question)

        # append the question and the role (user) as a message to the session state
        st.session_state.messages.append({"role": "user", "content": question})

        # respond as the assistant with the answer
        with st.chat_message("assistant"):

            # making sure there are no messages present when generating the answer
            message_placeholder = st.empty()

            # putting a spinning icon to show that the query is in progress
            with st.status("Searching...", expanded=False) as status:

                # passing the question into the kendra search function, which later invokes the llm
                if rag_option == "With RAG":
                    search_category = "all"

                    if search_category_option == "내부":
                        search_category = "internal"
                    elif search_category_option == "공개":
                        search_category = "public"
                    elif search_category_option == "기타":
                        search_category = "unclassified"

                    answer = chat_with_rag(question, region_name_selected, model_selected, search_category)
                else:
                    answer = chat(question, region_name_selected, model_selected)
                
                # writing the answer to the front end
                message_placeholder.markdown(f"{answer}")

                # showing a completion message to the front end
                status.update(label="Answered.", state="complete", expanded=False)

        # appending the results to the session state
        st.session_state.messages.append({"role": "assistant",
                                        "content": answer})