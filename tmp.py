
# -*- coding: utf-8 -*-
from pdfminer.high_level import extract_text
import re
# def simple_split_into_sentences(text):
#     """简单按行分割并尝试识别句子结束。"""
#     lines = text.split('\n')
#     sentences = []
#     current_sentence = ''
#     for line in lines:
#         # 假设句子结束于.!?后紧跟换行符或行尾
#         parts = re.split(r'(?<=[.!?])\s*', line)
#         for part in parts:
#             if part:  # 确保部分非空
#                 current_sentence += part
#                 if part[-1] in '.!?':  # 如果以句末符号结束
#                     sentences.append(current_sentence.strip())
#                     current_sentence = ''  # 重置当前句子
#         # 如果当前行没有以句末符号结束，但已经是最后一行，添加到句子列表
#         if line and not line.endswith(('。','!','?')) and line == lines[-1]:
#             sentences.append(current_sentence.strip())
#     return sentences

# def extract_sentences_from_pdf(pdf_path):
#     """
#     从PDF文件中提取所有文本，并尝试分割成一句一句的列表。
    
#     :param pdf_path: PDF文件的路径
#     :return: 一句话一个元素的列表
#     """
#     # 使用pdfminer提取整个PDF的文本
#     text = extract_text(pdf_path)
#     sentences = text.split('\n')
#     # 去除空字符串，防止因连续换行导致的空项
#     sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
#     # 尝试简单的句子分割
#     #sentences = simple_split_into_sentences(text)
#     return sentences

# 使用函数，提供PDF文件的路径
# pdf_file_path = r"E:\pythonProject\pytorch\ExtractInfo\test.pdf"
# text = extract_text(pdf_file_path)
# sentences_list = extract_sentences_from_pdf(pdf_file_path)

# 打印提取的句子列表
# print(sentences_list)
# print(f"该文档总共有{len(sentences_list)}个句子。")
def remove_specific_punctuation(text):
    """
    删除字符串中指定的标点符号，保留汉字和数字。
    这里以删除中英文常见标点为例。
    """
    # 定义需要删除的标点符号集合，包括中文和英文标点
    punctuation_to_remove = r'[.·\s]'
    
    # 使用正则表达式替换指定的标点符号为空字符串
    cleaned_text = re.sub(punctuation_to_remove, '', text)
    return cleaned_text

def split_into_sentences(text):
    # 使用换行符分割大文本为块
    blocks = text.split('\n')
    all_sentences = []  # 保存所有句子的列表
    sentences=""
    for block in blocks:
        if "。" in block:
            # 在每个块内部，进一步按句号分割
            sentences_in_block = block.split('。')
            for s in sentences_in_block:
                sentences=sentences+s
                sentences=remove_specific_punctuation(sentences)
                all_sentences.append(sentences)
                sentences=""
            sentences=sentences_in_block[-1]
        else:
            sentences=sentences+block
        
        # 去除空白字符串，避免因连续换行或句号导致的空元素
    all_sentences = [s.strip() for s in all_sentences if s.strip()]
        
        # # 将当前块中的所有有效句子添加到总列表中
        # all_sentences.extend(sentences_in_block)
    
    # 返回处理后的句子列表
    return all_sentences

# 示例文本
# text = """这是第\n
# 一句。这是第二句。这是第三句\n
# 的一部分。这是第三句的另一部分。\n
# 这是第四句。"""
# pdf_file_path = r"./test_ocr.pdf"

# text = extract_text(pdf_file_path)
# print(text)
# 调用函数并打印结果
# sentences_list = split_into_sentences(text)
# for i,s in enumerate(sentences_list):
#     print(f"句子{i+1}:",s)

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Model directory
model_dir = r"E:\pythonProject\pytorch\intelTextExtraction\pretrained_Bert"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)

# Define function to calculate similarity
def calc_similarity(s1, s2):
    # Tokenize input sentences
    inputs_1 = tokenizer(s1, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    inputs_2 = tokenizer(s2, return_tensors='pt', max_length=128, truncation=True, padding='max_length')

    # Get embeddings from BERT model
    with torch.no_grad():
        embeddings_1 = model(**inputs_1).last_hidden_state.mean(dim=1).numpy()
        embeddings_2 = model(**inputs_2).last_hidden_state.mean(dim=1).numpy()

    # Calculate cosine similarity
    sim = np.dot(embeddings_1[0], embeddings_2[0]) / (np.linalg.norm(embeddings_1[0]) * np.linalg.norm(embeddings_2[0]))
    return sim

# Example sentences
s1 = "结合掌子面地质情况，初步推测该段围岩岩性与掌子面情况大致相同，以粉质黏土为主，局部夹杂孤石，多呈散体状结构，整体结合差"
s2 = "雷达反射波范围内同相轴较连续，以中低频信号为主，振幅幅值一般，中部雷达信号振幅幅值较强且存在多次震荡现象。"

# Calculate similarity
similarity = calc_similarity(s1, s2)
print(f"Similarity: {similarity}")

