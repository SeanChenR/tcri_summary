import re
import os
import time
import json
import pandas as pd
from tqdm import tqdm
from datetime import date
from rouge_chinese import Rouge
from sentence_transformers import SentenceTransformer
from tcri_preprocessing import tcri_preprocessing
from LLM_Model import llama2, llama3, Llama3_FineTune_Model, Llama2_FineTune_Model, gpt4o, gpt4o_mini_finetune, gpt4o_combine_summary, tcri_rouge, tcri_sbert
from art_text import art_text

import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")

import warnings
warnings.filterwarnings('ignore')

def execute_time(start_time, end_time, string):
    execution_time = end_time - start_time
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{}耗時：{:0>2}時{:0>2}分{:05.2f}秒".format(string, int(hours), int(minutes), seconds))

# 處理關鍵字的指令
def tcri_keyword(file_path):
    keyword_data = pd.read_excel(file_path)

    category_dict = {}

    for index, row in keyword_data.iterrows():
        scode, keyword, instruction = row['小分類碼'], row['關鍵字'], row['指令']
        
        # 如果 scode 不存在 dict 中的話，會建立這個 keys
        if scode not in category_dict:
            category_dict[scode] = {}
        
        # 新增關鍵字和對應的指令在對應的 scode 底下
        category_dict[scode][keyword] = instruction
    
    return category_dict

# 處理正規表達式的指令
def tcri_regex(file_path):
    # 將XXX的部分替換為正規表達式的規則，並且一個正規表達式對應一個小分類碼
    data = pd.read_excel(file_path)
    no_keyword_data = data[data['關鍵字'] == '無關鍵字']
    no_keyword_data.reset_index(inplace=True)
    no_keyword_data.drop(columns="index", inplace=True)
    re_data = no_keyword_data[['小分類碼', '正規表達式', '指令']]
    re_data['正規表達式'].fillna("無關鍵字也無條件指令", inplace=True)
    regex = r'([a-zA-Z\u4e00-\u9fa50-9,._\-\s，、]+)'
    re_dict = {}

    # 每個小分類碼的正規表達式對應到相應的指令
    for index, row in re_data.iterrows():
        s_code, regex_text, command = row['小分類碼'], row['正規表達式'], row['指令']
        if regex_text == "無關鍵字也無條件指令":
            if s_code in re_dict:
                re_dict[s_code][regex_text] = command
            else:
                re_dict[s_code] = {regex_text: command}
        else:
            regex_text = regex_text.replace("XXX", regex)
            if s_code in re_dict:
                re_dict[s_code][regex_text] = command
            else:
                re_dict[s_code] = {regex_text: command}
    return re_dict

# 將整理好的關鍵字對應指令寫入 excel 檔
def tcri_keyword_update(summary_data, category_dict):
    keyword_list = []
    for scode, origin in zip(summary_data['小分類碼'], summary_data['資料清洗原文']):
        all_keyword = list(category_dict[scode].keys())
        for num, keyword_match in enumerate(all_keyword):
            if keyword_match in origin:
                keyword_list.append(keyword_match)
                break
            else:
                if num+1 == len(all_keyword):
                    keyword_list.append("無關鍵字")
                else:
                    pass

    summary_data['關鍵字'] = keyword_list
    summary_data['小分類碼'].fillna(value = "無小分類碼", inplace= True)
    return summary_data

# 將整理好的正規表達式對應指令寫入 excel 檔
def tcri_regex_update(summary_data, re_dict):
    regex_list = []
    for scode, origin in zip(summary_data['小分類碼'], summary_data['資料清洗原文']):
        all_re = list(re_dict[scode].keys())
        for num, re_match in enumerate(all_re):
            matches = re.findall(re_match, origin)
            if matches:
                regex_list.append(re_match)
                break
            else:
                if num+1 == len(all_re):
                    regex_list.append("無關鍵字也無條件指令")
                else:
                    pass
    summary_data['正規表達式'] = regex_list
    return summary_data

# 使用 GPT-4o 模型進行摘要
def tcri_run_summarize_gpt4o_model(summary_data, category_dict, regex_dict):
    start_time = time.time()
    gpt4o_chain = gpt4o()
    model, count = "gpt-4o", 1
    summary_result = []
    for (origin, s_code, keyword, regex_text, company) in tqdm(zip(summary_data['資料清洗原文'], summary_data['小分類碼'], summary_data['關鍵字'], summary_data['正規表達式'], summary_data['公司簡稱']), total=len(summary_data)):
        # 若沒有關鍵字才會去讀取正規表達式的指令
        if keyword == "無關鍵字":
            system_prompt = regex_dict[s_code][regex_text]
        else:
            system_prompt = category_dict[s_code][keyword]

        result = company + "公告" + gpt4o_chain.invoke({"system_prompt":system_prompt, "input_sentence":origin}).content
        print(result)
        summary_result.append(result)
        print(f"第{count}筆摘要完成！")
        count += 1

    summary_data['關鍵字摘要'] = summary_result
    end_time = time.time()

    execute_time(start_time, end_time, f"本次摘要使用{model}模型，完成共{len(summary_result)}筆摘要！")
    return summary_data, model

# 使用 GPT-4o-mini 的 FineTune 模型進行摘要
def tcri_run_summarize_gpt4omini_finetune_model(summary_data, category_dict, regex_dict):
    start_time = time.time()
    gpt4o_chain = gpt4o_mini_finetune()
    model, count = "gpt-4o-mini-finetune", 1
    summary_result = []
    for (origin, s_code, keyword, regex_text, company) in tqdm(zip(summary_data['資料清洗原文'], summary_data['小分類碼'], summary_data['關鍵字'], summary_data['正規表達式'], summary_data['公司簡稱']), total=len(summary_data)):
        # 若沒有關鍵字才會去讀取正規表達式的指令
        if keyword == "無關鍵字":
            system_prompt = regex_dict[s_code][regex_text]
        else:
            system_prompt = category_dict[s_code][keyword]

        result = company + "公告" + gpt4o_chain.invoke({"system_prompt":system_prompt, "input_sentence":origin}).content
        print(result)
        summary_result.append(result)
        print(f"第{count}筆摘要完成！")
        count += 1

    summary_data['關鍵字摘要'] = summary_result
    end_time = time.time()

    execute_time(start_time, end_time, f"本次摘要使用{model}模型，完成共{len(summary_result)}筆摘要！")
    return summary_data, model

# 使用台智雲 Llama2 模型進行摘要
def tcri_run_summarize_llama2_model(summary_data, category_dict, regex_dict):
    start_time = time.time()
    model, count = "llama2", 1
    summary_result = []

    for (origin, s_code, keyword, regex_text, company) in tqdm(zip(summary_data['資料清洗原文'], summary_data['小分類碼'], summary_data['關鍵字'], summary_data['正規表達式'], summary_data['公司簡稱']), total=len(summary_data)):
        # 若沒有關鍵字才會去讀取正規表達式的指令
        if keyword == "無關鍵字":
            command = regex_dict[s_code][regex_text]
        else:
            command = category_dict[s_code][keyword]

        system_prompt = f"你是一位擅長財報摘要的研究員，專門於對公司公告重訊進行摘要。\
                            我將提供一些原始的公告內容，請您將其摘要成一個簡潔的句子。\
                            請只要涵蓋關鍵訊息：例如{command}。"
        
        user_message = ["請將以下公告原文根據指令進行「中文摘要」，公告原文：" + origin]

        result = company + "公告" + llama2(system_prompt, user_message)
        print(result)
        summary_result.append(result)
        print(f"第{count}筆摘要完成！")
        count += 1

    summary_data['關鍵字摘要'] = summary_result
    end_time = time.time()
    execute_time(start_time, end_time, f"本次摘要使用{model}模型，完成共{len(summary_result)}筆摘要！")
    return summary_data, model

# 使用台智雲 Llama3 模型進行摘要
def tcri_run_summarize_llama3_model(summary_data, category_dict, regex_dict):
    start_time = time.time()
    model, count = "llama3", 1
    summary_result = []

    for (origin, s_code, keyword, regex_text, company) in tqdm(zip(summary_data['資料清洗原文'], summary_data['小分類碼'], summary_data['關鍵字'], summary_data['正規表達式'], summary_data['公司簡稱']), total=len(summary_data)):
        # 若沒有關鍵字才會去讀取正規表達式的指令
        if keyword == "無關鍵字":
            command = regex_dict[s_code][regex_text]
        else:
            command = category_dict[s_code][keyword]

        system_prompt = f"你是一位擅長財報摘要的研究員，專門於對公司公告重訊進行摘要。\
                            我將提供一些原始的公告內容，請您將其摘要成一個簡潔的句子。\
                            請只要涵蓋關鍵訊息：例如{command}。"
        
        user_message = ["請將以下公告原文根據指令進行「中文摘要」，公告原文：" + origin]

        result = company + "公告" + llama3(system_prompt, user_message)
        print(result)
        summary_result.append(result)
        print(f"第{count}筆摘要完成！")
        count += 1

    summary_data['關鍵字摘要'] = summary_result
    end_time = time.time()
    execute_time(start_time, end_time, f"本次摘要使用{model}模型，完成共{len(summary_result)}筆摘要！")
    return summary_data, model

# 使用台智雲 FineTune Llama2 模型進行摘要
def tcri_run_summarize_finetune2_model(summary_data, category_dict, regex_dict):
    start_time = time.time()
    model, count = "finetune-llama2", 1
    summary_result = []

    for (origin, s_code, keyword, regex_text, company) in tqdm(zip(summary_data['資料清洗原文'], summary_data['小分類碼'], summary_data['關鍵字'], summary_data['正規表達式'], summary_data['公司簡稱']), total=len(summary_data)):
        # 若沒有關鍵字才會去讀取正規表達式的指令
        if keyword == "無關鍵字":
            command = regex_dict[s_code][regex_text]
        else:
            command = category_dict[s_code][keyword]

        system_prompt = f"你是一位擅長財報摘要的研究員，專門於對公司公告重訊進行摘要。\
                            我將提供一些原始的公告內容，請您將其摘要成一個簡潔的句子。\
                            請只要涵蓋關鍵訊息：例如{command}。"
        
        user_message = ["請將以下公告原文根據指令進行「中文摘要」，公告原文：" + origin]
        num_tokens = len(tokenizer.encode(user_message[0]))
        if num_tokens > 4000:
            result = "此筆資料token數超過上限！"
        else:
            result = company + "公告" + Llama2_FineTune_Model(system_prompt, user_message)
        print(result)
        summary_result.append(result)
        print(f"第{count}筆摘要完成！")
        count += 1

    summary_data['關鍵字摘要'] = summary_result
    end_time = time.time()
    execute_time(start_time, end_time, f"本次摘要使用llama2 {model}模型，完成共{len(summary_result)}筆摘要！")
    return summary_data, model

# 使用台智雲 FineTune Llama3 模型進行摘要
def tcri_run_summarize_finetune3_model(summary_data, category_dict, regex_dict):
    start_time = time.time()
    model, count = "finetune-llama3", 1
    summary_result = []

    for (origin, s_code, keyword, regex_text, company) in tqdm(zip(summary_data['資料清洗原文'], summary_data['小分類碼'], summary_data['關鍵字'], summary_data['正規表達式'], summary_data['公司簡稱']), total=len(summary_data)):
        # 若沒有關鍵字才會去讀取正規表達式的指令
        if keyword == "無關鍵字":
            command = regex_dict[s_code][regex_text]
        else:
            command = category_dict[s_code][keyword]

        system_prompt = f"你是一位擅長財報摘要的研究員，專門於對公司公告重訊進行摘要。\
                            我將提供一些原始的公告內容，請您將其摘要成一個簡潔的句子。\
                            請只要涵蓋關鍵訊息：例如{command}。"
        
        user_message = ["請將以下公告原文根據指令進行「中文摘要」，公告原文：" + origin]
        num_tokens = len(tokenizer.encode(user_message[0]))
        if num_tokens > 8000:
            result = "此筆資料token數超過上限！"
        else:
            result = company + "公告" + Llama3_FineTune_Model(system_prompt, user_message)
        print(result)
        summary_result.append(result)
        print(f"第{count}筆摘要完成！")
        count += 1

    summary_data['關鍵字摘要'] = summary_result
    end_time = time.time()
    execute_time(start_time, end_time, f"本次摘要使用llama3 {model}模型，完成共{len(summary_result)}筆摘要！")
    return summary_data, model

# 處理相似度比對的函數
def tcri_similarity_comparison(sbert_model, rouge, summary_data, model):
    rouge_1_list, rouge_2_list, rouge_L_list, sbert_list = [], [], [], []
    if model == 'combine':
        artificial_summary, ai_summary = summary_data['人工摘要'], summary_data['合併摘要結果']
    else:
        artificial_summary, ai_summary = summary_data['摘要'], summary_data['關鍵字摘要']
    
    for i,j in zip(artificial_summary, ai_summary):
        # 使用 Rouge
        rouge_1_f1, rouge_2_f1, rouge_L_f1 = tcri_rouge(rouge, i, j)
        # 使用 Sentence Bert
        sbert_similarity = tcri_sbert(sbert_model, i, j)
        rouge_1_list.append(rouge_1_f1)
        rouge_2_list.append(rouge_2_f1)
        rouge_L_list.append(rouge_L_f1)
        sbert_list.append(sbert_similarity)

    summary_data["rouge-1_f1"] = rouge_1_list
    summary_data["rouge-2_f1"] = rouge_2_list
    summary_data["rouge-L_f1"] = rouge_L_list
    summary_data["sbert_similarity"] = sbert_list
    return summary_data

# 處理匯出輸出檔的函數，如果需要 Json 檔檢視關鍵字對應指令可以解除註解
def tcri_output(summary_data, model, category_dict, regex_dict):
    today = date.today()
    if model == 'gpt-4o':
        output_file = f"output_file/自動化摘要_{model}_{today}.xlsx"
    elif model == 'combine':
        output_file = f"output_file/自動化摘要_台智雲finetune-llama2&llama3合併_{today}.xlsx"
    else:
        output_file = f"output_file/自動化摘要_台智雲{model}_{today}.xlsx"
    summary_data.to_excel(output_file, index=False)

    # 若有需要解除註解，會將整理好的分類碼中對應的指令和涵正規表達法的指令輸出成 json 檔
    # keyword_json_path = f"output_file/指令彙總/關鍵字彙總_{today}.json"
    # regex_json_path = f"output_file/指令彙總/正規表達式_{today}.json"
    # with open(keyword_json_path, 'w', encoding='utf-8') as json_file:
    #     json.dump(category_dict, json_file, ensure_ascii=False, indent=4)
    # with open(regex_json_path, 'w', encoding='utf-8') as json_file:
    #     json.dump(regex_dict, json_file, ensure_ascii=False, indent=4)

    print(f"完成摘要並匯出檔案 -> {output_file}")

# 合併 llama2-fine-tune 和 llama3-fine-tune 的結果，使用 Openai 將其最後摘要一次
# 會判斷 output_file 資料夾在當日兩個模型是不是都有跑過並產出資料
def tcri_combine_summary(llama2_file_path, llama3_file_path):
    task = "combine"
    llama2_file = pd.read_excel(llama2_file_path)
    llama3_file = pd.read_excel(llama3_file_path)

    gpt4o_combine_chain = gpt4o_combine_summary()
    comnbine_result = []
    for llama2, llama3 in tqdm(zip(llama2_file['關鍵字摘要'], llama3_file['關鍵字摘要']), total=len(llama2_file['關鍵字摘要'])):
        if llama2 == "此筆資料token數超過上限！":
            comnbine_result.append(llama3)
            print(llama3)
        else:
            response = gpt4o_combine_chain.invoke({"llama2": llama2, "llama3": llama3})
            comnbine_result.append(response.content)
            print(response.content)

    combine_file = pd.DataFrame(columns=['llama2', 'llama3', '人工摘要','合併摘要結果'])
    combine_file['llama2'] = llama2_file['關鍵字摘要']
    combine_file['llama3'] = llama3_file['關鍵字摘要']
    combine_file['人工摘要'] = llama2_file['摘要']
    combine_file['合併摘要結果'] = comnbine_result
    return combine_file, task

if __name__ == "__main__":
    preprocessor = tcri_preprocessing()

    keyword_re_file_path = "input_file/關鍵字彙總(1017).xlsx"
    summary_file_path = "input_file/自動化摘要_10071009.xlsx"

    summary_data = pd.read_excel(summary_file_path)
    start_time = time.time()
    clean_list = []
    for i in summary_data['說明']:
        final_text = preprocessor.main(i)
        clean_list.append(final_text)

    summary_data['資料清洗原文'] = clean_list
    end_time = time.time()
    execute_time(start_time, end_time, "資料清洗完成")

    category_dict = tcri_keyword(keyword_re_file_path)
    regex_dict = tcri_regex(keyword_re_file_path)
    summary_data_update = tcri_keyword_update(summary_data, category_dict)
    summary_data_regex = tcri_regex_update(summary_data_update, regex_dict)
    select_model = input("請選擇模型或任務(ffm-llama2/ffm-llama3/ffm-finetune2/ffm-finetune3/gpt-4o/gpt-4o-mini-finetune/combine-summary)：")
    if select_model == "ffm-llama3":
        art_text.llama3_text()
        print(f"你選擇{select_model}作為摘要模型！")
        summary_result, model = tcri_run_summarize_llama3_model(summary_data_regex, category_dict, regex_dict)
    elif select_model == "ffm-llama2":
        art_text.llama2_text()
        print(f"你選擇{select_model}作為摘要模型！")
        summary_result, model = tcri_run_summarize_llama2_model(summary_data_regex, category_dict, regex_dict)
    elif select_model == "ffm-finetune2":
        art_text.finetune_llama2_text()
        print(f"你選擇{select_model}作為摘要模型！")
        summary_result, model = tcri_run_summarize_finetune2_model(summary_data_regex, category_dict, regex_dict)
    elif select_model == "ffm-finetune3":
        art_text.finetune_llama3_text()
        print(f"你選擇{select_model}作為摘要模型！")
        summary_result, model = tcri_run_summarize_finetune3_model(summary_data_regex, category_dict, regex_dict)
    elif select_model == "gpt-4o":
        art_text.openai_gpt_4o_text()
        print(f"你選擇{select_model}作為摘要模型！")
        summary_result, model = tcri_run_summarize_gpt4o_model(summary_data_regex, category_dict, regex_dict)
    elif select_model == "gpt-4o-mini-finetune":
        art_text.openai_gpt_4o_mini_text()
        print(f"你選擇{select_model}作為摘要模型！")
        summary_result, model = tcri_run_summarize_gpt4omini_finetune_model(summary_data_regex, category_dict, regex_dict)
    elif select_model == "combine-summary":
        # 設定檔案路徑
        today = date.today()
        llama2_file_path = f"output_file/自動化摘要_台智雲finetune-llama2_{today}.xlsx"
        llama3_file_path = f"output_file/自動化摘要_台智雲finetune-llama3_{today}.xlsx"

        if os.path.exists(llama2_file_path) and os.path.exists(llama3_file_path):
            art_text.openai_combine_summary()
            print(f"進行{select_model}任務，模型為 GPT-4o！")
            summary_result, model = tcri_combine_summary(llama2_file_path, llama3_file_path)
        else:
            raise OSError("其中一個微調模型尚未完成摘要，請先進行 Fine Tune Llama2 和 Fine Tune Llama3 再接著進行此任務！")
    else:
        raise ValueError("Please select a model from the menu: ffm-llama2/ffm-llama3/ffm-finetune2/ffm-finetune3/gpt-4o/combine-summary")
    
    try:
        rouge = Rouge()
        sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        summary_result_with_score = tcri_similarity_comparison(sbert_model, rouge, summary_result, model)
    except:
        summary_result_with_score = summary_result
    tcri_output(summary_result_with_score, model, category_dict, regex_dict)
