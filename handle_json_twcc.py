import json
import pandas as pd
import tiktoken
from langchain_core.prompts import PromptTemplate

data = pd.read_excel("2020to20240331.xlsx")

filt = (data['出貨(Y/N/D)'] == 'Y')

df = data.loc[filt, ['公司簡稱', '摘要', '說明']]

df.dropna(inplace=True)

df.rename(columns={'公司簡稱': 'company', '摘要': 'summarization', '說明': 'original'}, inplace=True)

system_prompt = {
    "role":"system",
    "content":"你是一個信用評等研究員，要對公司公開資訊觀測站公告重訊進行摘要，請僅摘錄重要資訊，如該事件之發生原因、因應之措施、交易之標的物為何、交易之對象、交易之數量、交易之金額，如該項內容為「不適用」請勿摘錄，請不要使用條列式表達，請將摘錄內容連成一句話。"
}

human_prompt = (
    PromptTemplate.from_template('"role":"human", "content":"{origin}"')
)

ai_prompt = (
    PromptTemplate.from_template('"role":"assistant", "content":"{summarize}"')
)

tokenizer = tiktoken.get_encoding("cl100k_base")

finetune_data = []
for (company, summarization, original) in zip(df['company'], df['summarization'], df['original']):
    summarization = summarization.replace('”', "'").replace('“', "'").replace('\n', '').replace('\t', '').replace('"', "'").replace('\\', '')
    original = original.replace('”', "'").replace('“', "'").replace('\n', '').replace('\t', '').replace('"', "'").replace('\\', '')
    if company in original:
        human = human_prompt.format(origin=original)
        final_human_prompt = "{" + human + "}"
        finetune_human = [json.loads(final_human_prompt)]
        AI = ai_prompt.format(summarize=summarization)
        final_ai_prompt = "{" + AI + "}"
        finetune_ai = [json.loads(final_ai_prompt)]
        final_data = [system_prompt] + finetune_human + finetune_ai
        finetune_data.append(final_data)
    else:
        original_with_company = company + original
        human = human_prompt.format(origin=original_with_company)
        final_human_prompt = "{" + human + "}"
        finetune_human = [json.loads(final_human_prompt)]
        AI = ai_prompt.format(summarize=summarization)
        final_ai_prompt = "{" + AI + "}"
        finetune_ai = [json.loads(final_ai_prompt)]
        final_data = [system_prompt] + finetune_human + finetune_ai
        finetune_data.append(final_data)

with open('summarize_2020to20240331.jsonl', 'w', encoding='utf-8') as file:
    for entry in finetune_data:
        num_tokens = len(tokenizer.encode(str(entry)))
        if num_tokens >= 4096:
            continue
        else:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

print("Finish!")