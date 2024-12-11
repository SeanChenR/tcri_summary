# both
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

# TWCC package
import json
import requests

# gpt4o package
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts.prompt import PromptTemplate

# rouge & sbert
import jieba
from sentence_transformers import util

openai_model = config['openai']['MODEL_NAME']
openai_fine_tune_model = config['openai']['FineTune_MODEL_NAME']
openai_api_key = config['openai']['API_KEY']
openai.api_key = openai_api_key

# parameters
max_new_tokens = 500
temperature = 0.01
top_k = 10
top_p = 1.0
frequence_penalty = 1.0

llama3_MODEL_NAME = config['TWSC_public']['Llama3_MODEL_NAME']
llama2_MODEL_NAME = config['TWSC_public']['Llama2_MODEL_NAME']
TWSC_public_API_KEY = config['TWSC_public']['API_KEY']
TWSC_public_API_URL = config['TWSC_public']['API_URL']
TWSC_public_API_HOST = config['TWSC_public']['API_HOST']

# KEY & URL 需要每次到congfig.ini修改，按時間計費，用完Model記得關閉
Llama3_FT_MODEL_NAME = config['FineTune_Model']['Llama3_MODEL_NAME']
Llama3_FT_API_KEY = config['FineTune_Model']['Llama3_API_KEY']
Llama3_FT_API_URL = config['FineTune_Model']['Llama3_API_URL']
Llama2_FT_MODEL_NAME = config['FineTune_Model']['Llama2_MODEL_NAME']
Llama2_FT_API_KEY = config['FineTune_Model']['Llama2_API_KEY']
Llama2_FT_API_URL = config['FineTune_Model']['Llama2_API_URL']

def gpt4o():
    prompt = """
            # CONTEXT #
            你將會接收到一段文字, 該段文字會被涵蓋在html的tag之中,
            看到<text>代表句子的開始,看到</text>代表句子的結束，根據# OBJECTIVE #指令來做摘要

            # OBJECTIVE #
            {system_prompt}

            # RESPONSE #
            回覆的內容只要有你的摘要結果就好

            <text>
            {input_sentence}
            </text>
            """

    prompt_template = PromptTemplate.from_template(prompt)
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model_name=openai_model
    )
    chain = prompt_template | llm
    return chain

def gpt4o_mini_finetune():
    prompt = """
            # CONTEXT #
            你將會接收到一段文字, 該段文字會被涵蓋在html的tag之中,
            看到<text>代表句子的開始,看到</text>代表句子的結束，根據# OBJECTIVE #指令來做摘要

            # OBJECTIVE #
            {system_prompt}

            # RESPONSE #
            回覆的內容只要有你的摘要結果就好

            <text>
            {input_sentence}
            </text>
            """

    prompt_template = PromptTemplate.from_template(prompt)
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model_name=openai_fine_tune_model
    )
    chain = prompt_template | llm
    return chain

def gpt4o_combine_summary():
    prompt = """
            # CONTEXT #
            你將會接收到兩段文字, 該段文字會被涵蓋在html的tag之中,
            看到<text1>, <text2>代表句子的開始,看到</text1>, </text2>代表句子的結束，根據# OBJECTIVE #指令來做摘要

            # OBJECTIVE #
            <text1></text1>, <text2></text2>中為某一段文章的摘要，如為董事放棄認購相關摘要，請勿修改任何文字，其他請將重複的文字刪除，並修飾成通順的句子

            # RESPONSE #
            回覆的內容只要有你的摘要結果就好

            <text1>
            {llama2}
            </text1>

            <text2>
            {llama3}
            </text2>
            """

    prompt_template = PromptTemplate.from_template(prompt)
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0,
        model_name=openai_model
    )
    chain = prompt_template | llm
    return chain

def llama2(system, contents):
    headers = {
        "content-type": "application/json", 
        "X-API-KEY": TWSC_public_API_KEY,
        "X-API-HOST": TWSC_public_API_HOST}
    
    roles = ["human", "assistant"]
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    for index, content in enumerate(contents):
        messages.append({"role": roles[index % 2], "content": content})
    data = {
        "model": llama2_MODEL_NAME,
        "messages": messages,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "frequence_penalty": frequence_penalty
        }
    }

    result = ""
    try:
        response = requests.post(TWSC_public_API_URL + "/models/conversation", json=data, headers=headers)
        if response.status_code == 200:
            result = json.loads(response.text, strict=False)['generated_text']
        else:
            print("error")
    except:
        print("error")
    return result.strip("\n")

def llama3(system, contents):
    headers = {
        "content-type": "application/json", 
        "X-API-KEY": TWSC_public_API_KEY,
        "X-API-HOST": TWSC_public_API_HOST}
    
    roles = ["human", "assistant"]
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    for index, content in enumerate(contents):
        messages.append({"role": roles[index % 2], "content": content})
    data = {
        "model": llama3_MODEL_NAME,
        "messages": messages,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "frequence_penalty": frequence_penalty
        }
    }

    result = ""
    try:
        response = requests.post(TWSC_public_API_URL + "/models/conversation", json=data, headers=headers)
        if response.status_code == 200:
            result = json.loads(response.text, strict=False)['generated_text']
        else:
            print("error")
    except:
        print("error")
    return result.strip("\n")

def Llama3_FineTune_Model(system, contents):
    headers = {
        "content-type": "application/json", 
        "X-API-Key": Llama3_FT_API_KEY}
    
    roles = ["human", "assistant"]
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    for index, content in enumerate(contents):
        messages.append({"role": roles[index % 2], "content": content})
    data = {
        "model": Llama3_FT_MODEL_NAME,
        "messages": messages,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "frequence_penalty": frequence_penalty
        }
    }

    result = ""
    try:
        response = requests.post(Llama3_FT_API_URL + "/api/models/conversation", json=data, headers=headers)
        if response.status_code == 200:
            result = json.loads(response.text, strict=False)['generated_text']
        else:
            print("error")
    except:
        print("error")
    return result.strip("\n")

def Llama2_FineTune_Model(system, contents):
    headers = {
        "content-type": "application/json", 
        "X-API-Key": Llama2_FT_API_KEY}
    
    roles = ["human", "assistant"]
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    for index, content in enumerate(contents):
        messages.append({"role": roles[index % 2], "content": content})
    data = {
        "model": Llama2_FT_MODEL_NAME,
        "messages": messages,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "frequence_penalty": frequence_penalty
        }
    }

    result = ""
    try:
        response = requests.post(Llama2_FT_API_URL + "/api/models/conversation", json=data, headers=headers)
        if response.status_code == 200:
            result = json.loads(response.text, strict=False)['generated_text']
        else:
            print("error")
    except:
        print("error")
    return result.strip("\n")

def tcri_rouge(rouge, artificial, ai):
    artificial = ' '.join(jieba.cut(artificial))
    ai = ' '.join(jieba.cut(ai))

    rouge = rouge
    scores = rouge.get_scores(ai, artificial)
    try:
        score = scores[0]
    except:
        raise Exception("Rouge score error")
    rouge_1_f1 = score["rouge-1"]["f"]
    rouge_2_f1 = score["rouge-2"]["f"]
    rouge_L_f1 = score["rouge-l"]["f"]
    return rouge_1_f1, rouge_2_f1, rouge_L_f1

def tcri_sbert(sbert_model, artificial, ai):
    embedding_artificial = sbert_model.encode(artificial, convert_to_tensor=True)
    embedding_ai = sbert_model.encode(ai, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding_artificial, embedding_ai)
    return similarity.item()