# TCRI 自動化摘要

## python 環境 與 所需套件
![Static Badge](https://img.shields.io/badge/3.11-F7DF1E?style=for-the-badge&logo=python&label=Python&labelColor=black)

```python
pip install -r requirements.txt
```
## 執行py檔
```python
python tcri_summary_main.py
```

## 預設檔案
以下檔案若要更新或修改，重新上傳之餘，也記得更新程式碼內的檔案名稱。
| 檔案內容 (附檔名)     | 檔案名稱                   | 存放路徑             | 輸入輸出     | 描述                |
| -------------- | ---------------------- | ---------------- | -------- | ----------------- |
| 指令 (xlsx)      | 關鍵字彙總(0815).xlsx       | input_file       | 輸入       | 包含關鍵字和正規表達式下搭配的指令 |
| 摘要 (xlsx)      | 自動化摘要_07290731.xlsx    | input_file       | 輸入       | 要拿來摘要的檔案          |
| 關鍵字指令 (json)   | 關鍵字彙總(當天日期).json       | output_file/指令彙總 | 輸出（可不輸出） | 每個小分類碼底下的關鍵字指令    |
| 正規表達式指令 (json) | 正規表達式(當天日期).json       | output_file/指令彙總 | 輸出（可不輸出） | 每個小分類碼底下的正規表達式指令  |
| 摘要結果 (xlsx)    | 自動化摘要(模型名稱)(當天日期).xlsx | output_file      | 輸出       | 摘要輸出結果與摘要分數       |

## 選擇特定任務或模型
在執行主程式時，需要選擇摘要模型名稱，或是合併摘要的任務。

![2024-06-22 23.38.57](https://hackmd.io/_uploads/BJMZBO4LR.gif)

執行後可以看到，輸入對應的模型，就會用那個模型下去跑摘要的結果。

## Model Name, API KEY, Endpoint URL
若要更改 Model Name, API KEY, Endpoint URL 的話需要進入 config.ini 檔更改。基本上是只有 FineTune_Model 底下的有可能會隨著每次開啟模型而更改，其他是不太需要更改的，除非更換台智雲或 OpenAI 帳號。

## 台智雲模型注意事項
若使用的是 `ffm-llama3` 或 `ffm-llama2` 模型的話，key & url可以不用改，Model 可以隨台智雲更新滾動式調整欲使用之模型。但是若是要使用 finetune 模型的話，要先到台智雲 AFS 入口啟動模型並取回對應的 api & url，摘要結束後再回到 AFS 把模型關掉。

## OpenAI 模型注意事項
目前模型皆選擇 `GPT-4o` 模型，可以從執行 .py 檔，也可以使用 .ipynb 檔在 Colab 執行。若需要更換模型，需到 LLM_Model.py 中，找到 `gpt4o` 和 `gpt4o_combine_summary` 兩個函數中更改函數名字。