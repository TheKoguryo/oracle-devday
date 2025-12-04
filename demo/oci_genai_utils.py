import oci

import sys
import oracledb
import os

from langchain_oracledb.vectorstores import oraclevs
from langchain_oracledb.vectorstores import OracleVS

from langchain_oci import OCIGenAIEmbeddings
from langchain_oci import ChatOCIGenAI
from langchain_oci import ChatOCIOpenAI

import oci_openai
from oci_openai import (
    OciInstancePrincipalAuth,
    OciResourcePrincipalAuth,
    OciSessionAuth,
)

from langchain_community.vectorstores.utils import DistanceStrategy
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import configparser

from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from functools import lru_cache

config = None
signer = None
generative_ai_models = None
genai_available_regions = [ "sa-saopaulo-1", "eu-frankfurt-1", "ap-osaka-1", "uk-london-1", "us-chicago-1" ]
conn = None
COMPARTMENT_ID = None
AUTH_TYPE = None
endpoint = None
vector_store = None
VECTORSTORE_TABLENAME = "ORAVS_DOCUMENTS"

def initialize():
    print("Initializing program...")
    # init

    properties = configparser.ConfigParser()
    properties.read("app.env")

    DB_TYPE             = properties["DATABASE"]["TYPE"]
    DB_USERNAME         = properties["DATABASE"]["USERNAME"]
    DB_PASSWORD         = properties["DATABASE"]["PASSWORD"]
    DB_DSN              = properties["DATABASE"]["DSN"]
    DB_WALLET_LOCATION  = properties["DATABASE"]["WALLET_LOCATION"]
    DB_WALLET_PASSWORD  = properties["DATABASE"]["WALLET_PASSWORD"]

    global CONFIG_PROFILE
    CONFIG_PROFILE  = properties["OCI"]["CONFIG_PROFILE"]
    global COMPARTMENT_ID
    COMPARTMENT_ID  = properties["OCI"]["COMPARTMENT_ID"]
    global REGION
    REGION          = properties["OCI"]["REGION"]
    global AUTH_TYPE
    AUTH_TYPE       = properties["OCI"]["AUTH_TYPE"]

    global config
    global signer
    if AUTH_TYPE == "INSTANCE_PRINCIPAL":
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        config = {'region': REGION, 'tenancy': signer.tenancy_id}
    else:        
        config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)
        config['region'] = REGION

    try:
        global conn

        if DB_TYPE.startswith("ADB"):
            conn=oracledb.connect(
                config_dir=DB_WALLET_LOCATION,
                user=DB_USERNAME,
                password=DB_PASSWORD,
                dsn=DB_DSN,
                wallet_location=DB_WALLET_LOCATION,
                wallet_password=DB_WALLET_PASSWORD)
            
            print("Oracle 26ai Autonomous Database Connection successful!")
        else:
            conn = oracledb.connect(user=DB_USERNAME, password=DB_PASSWORD, dsn=DB_DSN)
            
            print("Connection successful!")
    except Exception as e:
        print("Database Connection failed!")
        print(e)
        sys.exit(1)

@lru_cache(maxsize=128)
def list_regions(): 
    print("list_regions")
    if AUTH_TYPE == "INSTANCE_PRINCIPAL":
        identity_client = oci.identity.IdentityClient(config, signer=signer)
    else:
        identity_client = oci.identity.IdentityClient(config)

    list_region_subscriptions_response = identity_client.list_region_subscriptions(tenancy_id=config['tenancy'])

    regions = []

    for region in list_region_subscriptions_response.data:
        if region.region_name in genai_available_regions:
            regions.append(region)

    sorted_regions = sorted(regions, key=lambda k: k.region_name)
    region_names = [region.region_name for region in sorted_regions]

    change_region(region_names[0])

    return region_names, sorted_regions

@lru_cache(maxsize=128)
def list_chat_models(region_name):
    print("list_chat_models")
    config['region'] = region_name

    if AUTH_TYPE == "INSTANCE_PRINCIPAL":
        generative_ai_client = oci.generative_ai.GenerativeAiClient(config, signer=signer)
    else:
        generative_ai_client = oci.generative_ai.GenerativeAiClient(config)

    list_models_response = generative_ai_client.list_models(
        compartment_id=COMPARTMENT_ID,
        limit=100)
    
    #print(list_models_response.data)
    
    models = [];

    for item in list_models_response.data.items:
        if len(item.capabilities) == 1 and "CHAT" in item.capabilities and item.time_deprecated is None and item.vendor != 'openai':
            models.append(item)

    sorted_models = sorted(models, key=lambda k: k.display_name, reverse=True)
    model_names = [model.display_name for model in sorted_models]
    
    global generative_ai_models
    generative_ai_models = sorted_models

    return model_names, generative_ai_models

def change_region(new_region_name):
    global region_name
    region_name = new_region_name

    print("region_name is updated to " + region_name)

    _init_vector_store(new_region_name)

def _init_vector_store(region_name):
    # Service endpoint
    global endpoint    
    endpoint = "https://inference.generativeai." + region_name + ".oci.oraclecloud.com"

    # 단계 3: EMBED(임베딩 생성)
    embeddings = OCIGenAIEmbeddings(
        model_id="cohere.embed-v4.0",
        service_endpoint=endpoint,
        compartment_id=COMPARTMENT_ID,
        auth_type=AUTH_TYPE,
        #auth_profile=CONFIG_PROFILE
    )

    # 단계 4: STORE(벡터스토어 생성)
    global vector_store
    vector_store = OracleVS(
        client=conn,        
        table_name=VECTORSTORE_TABLENAME,
        embedding_function=embeddings,
        distance_strategy=DistanceStrategy.COSINE,
    )

def chat(user_question, region_name, model):
    # Service endpoint
    endpoint = "https://inference.generativeai." + region_name + ".oci.oraclecloud.com"

    model_vendor = model.vendor
    model_id = model.id    
    print(model_vendor)
    print(model_id)

    # 단계 1: LLM(언어모델 생성)
    if model_vendor == 'openai':
        llm = ChatOCIOpenAI(
            auth=OciInstancePrincipalAuth(),
            compartment_id=COMPARTMENT_ID,
            model=model_id,
            service_endpoint=endpoint,
            store=False
        )
    else:
        llm = ChatOCIGenAI(
            model_id=model_id,
            service_endpoint=endpoint,
            compartment_id=COMPARTMENT_ID,
            provider=model_vendor,
            model_kwargs={"temperature": 0},
            auth_type=AUTH_TYPE,
            #auth_profile=CONFIG_PROFILE
        )

    # 단계 2: LLM(언어모델 생성)에 질의
    response = llm.invoke(user_question)

    print(f"# 답변: {response.content}")

    response_text = response.content

    return response_text


def chat_with_rag(user_question, region_name, model, search_category):
    # Service endpoint
    endpoint = "https://inference.generativeai." + region_name + ".oci.oraclecloud.com"

    model_vendor = model.vendor
    model_id = model.id    
    print(model_vendor)
    print(model_id)
    print(user_question)
    print(search_category)

    if vector_store is None:
        _init_vector_store(region_name)

    if search_category != "all":
        filter_dict = {"category": search_category};

        retrieved_docs = vector_store.similarity_search(user_question, 10, filter=filter_dict)
    else:
        retrieved_docs = vector_store.similarity_search(user_question, 10)

    print(len(retrieved_docs))
    for index, doc in enumerate(retrieved_docs):
        print(f"#### index: {index}")
        print(f"     metadata: {doc.metadata}")
        print(f"     page_content: {doc.page_content}")

    context = "\n\n".join([
        f"metadata: {doc.metadata}, page_content: {doc.page_content}"
        for doc in retrieved_docs
    ])
    question = user_question

    prompt = f"""You are an assistant for question-answering tasks. 
    You **must** answer **only** based on the provided context. 
    If the answer is not contained in the context, you **must** immediately say that you don't know, and you **must not** provide any additional explanation. 
    Answer in Korean.

    Your answer should:
    1. Provide a clear and detailed explanation in Korean **based only on the provided context**. 
    2. Be easy to understand with enough context and examples if relevant.
    3. **Only if the information is present in the provided context**: After the answer, include the source document names (and page numbers if available) in a **new line**, 
    following this format: [출처: 문서명, p.번호]. If the page number is not available, just include the document name.  
    **If the information is not in the context, ignore this item.**

    #Question: 
    {question} 
    #Context: 
    {context} 

    #Answer:
    """    

    # 단계 8: LLM(언어모델 생성)
    llm = ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=endpoint,
        compartment_id=COMPARTMENT_ID,
        provider=model_vendor,
        model_kwargs={"temperature": 0},
        auth_type=AUTH_TYPE,
        #auth_profile=CONFIG_PROFILE
    )

    response = llm.invoke(prompt)

    print(f"# user_question: {user_question}")
    print(f"# response: {response.content}")    

    response_text = response.content

    return response_text

def load_to_vector_store(file_path, original_filename, category):
    #_get_documents_by_source(original_filename)    

    print(f"✅ file_path: {file_path}")
    print(f"✅ original_filename: {original_filename}")
    print(f"✅ category: {category}")

    # 단계 0: 기존 문서 삭제
    # ✅ DELETE 실행
    cursor = conn.cursor()    
    delete_query = "DELETE FROM " + VECTORSTORE_TABLENAME + " WHERE JSON_VALUE(metadata, '$.source') = :source"
    cursor.execute(delete_query, {"source": original_filename})

    # ✅ 삭제된 행 개수 출력
    deleted_rows = cursor.rowcount
    print(f"✅ 삭제된 행 수: {deleted_rows}")

    # 단계 1: LOAD(문서 로드)
    #loader = PyPDFLoader(file_path)
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    if original_filename is not None:
        for doc in docs:
            doc.metadata["source"] = original_filename
            doc.metadata["category"] = category

    print(f"# 원문서 페이지수: {len(docs)}")

    # 단계 2: SPLIT(문서 분할)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)


    print(f"# 분할된 청크의수: {len(all_splits)}")
    #print(f"Chunk-0: {all_splits[0]}") # content

    # 단계 3: EMBED(임베딩 생성)
    embeddings = OCIGenAIEmbeddings(
        model_id="cohere.embed-v4.0",
        service_endpoint=endpoint,
        compartment_id=COMPARTMENT_ID,
        auth_type=AUTH_TYPE,
        #auth_profile=CONFIG_PROFILE
    )

    # 단계 4: STORE(벡터스토어에 문서 추가)
    result = vector_store.add_documents(all_splits)

    print(len(result))

def _get_documents_by_source(source_filename):
    print("####")
    cursor = conn.cursor()
    cursor.execute("select * from " + VECTORSTORE_TABLENAME + " where JSON_VALUE(metadata, '$.source') = '" + source_filename + "'")
    num_rows = 10
    while True:
        rows = cursor.fetchmany(size=num_rows)
        if not rows:
            break
        for row in rows:
            raw_data = row[0]
            metadata = row[2]
            print(raw_data.hex().upper())
            print(metadata)
    print("####")      

initialize()  # 스크립트 실행 시 자동으로 실행됨