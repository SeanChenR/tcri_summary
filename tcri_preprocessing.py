import re
import pandas as pd
from opencc import OpenCC
from regular_expression import half2full_list

# 西元 -> 民國
def convert_year_to_minguo(text):
    # 使用正規表達式提取數字
    year_match = re.search(r'\d{4}', text)
    if year_match:
        year = int(year_match.group())
        # 將西元年轉換成民國年
        minguo_year = year - 1911
        # 將民國年插入原字符串中
        result = re.sub(r'\d{4}', str(minguo_year), text, count=1)
        return result
    else:
        return text

class tcri_preprocessing:
    def __init__(self):
        self.cc = OpenCC('s2tw')

    # 簡體中文 -> 繁體中文
    def traditional_chinese(self, text):
        return self.cc.convert(text)

    # 民國 -> 西元
    def ROC2AD(self, text):
        pattern_slash = r'(\d{3,4})/(\d{1,2})/(\d{1,2})'
        pattern_yy = r'(\d{3,4})年'
        pattern_yy_with_txt = r'民國(\d{3})年'
        pattern_yymm = r'(\d{3,4})年(\d{1, 2})月'
        pattern_yymm_with_txt = r'民國(\d{3,4})年(\d{1,2})月'
        pattern_yymmdd = r'(\d{3,4})年(\d{1,2})月(\d{1,2})日'
        pattern_yymmdd_with_txt = r'民國(\d{3,4})年(\d{1,2})月(\d{1,2})日'
        pattern_yy2yy = r'(\d{3,4})至(\d{3,4})年'
        pattern_yy2yy_with_txt = r'民國(\d{3,4})至(\d{3,4})年'

        def repalce_slash(match):
            year, month, day = match.groups()
            if int(year) < 1911:
                year = int(year) + 1911
            return f"{year}/{month}/{day}"

        def replace_yymmdd_with_txt(match):
            year, month, day = map(int, match.groups())
            if year < 1911:
                year += 1911
            return f'{year}年{month}月{day}日'

        def replace_yymmdd(match):
            year, month, day = map(int, match.groups())
            if year < 1911:
                year += 1911
            return f'{year}年{month}月{day}日'

        def replace_yymm_with_txt(match):
            year, month = map(int, match.groups())
            if year < 1911:
                year += 1911
            return f'{year}年{month}月'

        def replace_yymm(match):
            year, month = map(int, match.groups())
            if year < 1911:
                year += 1911
            return f'{year}年{month}月'

        def replace_yy_with_txt(match):
            year = int(match.group(1))
            if year < 1911:
                year += 1911
            return f'{year}年'

        def replace_yy(match):
            year = int(match.group(1))
            if year < 1911:
                year += 1911
            return f'{year}年'

        def replace_yy2yy(match):
            year1, year2 = map(int, match.groups())
            if year1 < 1911:
                year1 += 1911
            if year2 < 1911:
                year2 += 1911
            return f'{year1}至{year2}年'
        
        def replace_yy2yy_with_txt(match):
            year1, year2 = map(int, match.groups())
            if year1 < 1911:
                year1 += 1911
            if year2 < 1911:
                year2 += 1911
            return f'{year1}至{year2}年'

        text = re.sub(pattern_slash, repalce_slash, text)
        text = re.sub(pattern_yymmdd_with_txt, replace_yymmdd_with_txt, text)
        text = re.sub(pattern_yymmdd, replace_yymmdd, text)
        text = re.sub(pattern_yymm_with_txt, replace_yymm_with_txt, text)
        text = re.sub(pattern_yymm, replace_yymm, text)
        text = re.sub(pattern_yy_with_txt, replace_yy_with_txt, text)
        text = re.sub(pattern_yy, replace_yy, text)
        text = re.sub(pattern_yy2yy_with_txt, replace_yy2yy_with_txt, text)
        text = re.sub(pattern_yy2yy, replace_yy2yy, text)
        return text

    # 半形 -> 全形
    def half2full(self, text):
        global half2full_list
        for condition in half2full_list:
            condition = condition.replace('XXX', '\d+')
            try:
                handle_text = re.findall(condition, text)[0]
                minguo_text = convert_year_to_minguo(handle_text)
                replace_text = minguo_text.replace('1', '１').replace('2', '２').replace('3', '３').replace('4', '４').replace('5', '５').replace('6', '６').replace('7', '７').replace('8', '８').replace('9', '９').replace('0', '０')
                final_text = text.replace(handle_text, replace_text)
                break
            except:
                final_text = text
        return final_text
    
    # 將國內轉成台灣，還有全形轉半形
    def full2half(self, text):
        final_text = text.replace("（", "(").replace("）", ")")
        return final_text

    # 將 anntype.xlsx 所出現的小分類碼與名稱刪除，也將不適用移除
    def anntype(self, text):
        anntype_data = pd.read_excel("input_file/anntype.xlsx")
        anntype_category = anntype_data['Category'].tolist()
        for category in anntype_category:
            if category in text:
                text = text.replace(category, "").replace("不適用", "").replace("本公司", "公司")
            else:
                text = text.replace("不適用", "").replace("本公司", "公司")
        return text
    
    def main(self, text):
        text_1 = self.ROC2AD(text)
        text_2 = self.anntype(text_1)
        text_3 = self.traditional_chinese(text_2)
        text_4 = self.half2full(text_3)
        final_text = self.full2half(text_4)
        return final_text


if __name__ == '__main__':
    processor = tcri_preprocessing()

    file_path = "資料清洗測試.xlsx"
    data = pd.read_excel(file_path)

    new_list = []
    for i in data['說明']:
        text_1 = processor.ROC2AD(i)
        text_2 = processor.anntype(text_1)
        text_3 = processor.traditional_chinese(text_2)
        text_4 = processor.half2full(text_3)
        text_5 = processor.full2half(text_4)
        new_list.append(text_5)

    data['資料清洗原文'] = new_list
    data.to_excel(file_path, index=False)
    print("完成！")