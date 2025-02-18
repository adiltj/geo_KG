# -*- coding: utf-8 -*-
##使用python38 版本
import sys
sys.path.append('./')
from model import BertForSentenceClassification
# from utils import logger_init
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader
import logging
import os
import sys
from gooey import Gooey, GooeyParser
from docx import Document
from data_loader import NERDataset
from data_loader import LoadDataset
from metrics import get_entities
import config
from config import SentenceClassificationModelConfig
from model import BertNER
# from riskEvaluation import riskEvaluation
from pdfminer.high_level import extract_text
import re

import warnings
# 修复打包后无法输出结构的问题
import codecs
import sys

if sys.stdout.encoding != 'UTF-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'UTF-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    

warnings.filterwarnings('ignore')
log_path = os.path.join('./logs/', 'logs_06-18.txt')
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(log_path),
                                logging.StreamHandler(sys.stdout)]
                    )


def SentenceClassification(config,texts):
    model = BertForSentenceClassification(config,
                                          config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行预测......")
    model = model.to(config.device)
    data_loader = LoadDataset(vocab_path=config.vocab_path,
                              tokenizer=BertTokenizer.from_pretrained(
                                  config.pretrained_model_dir).tokenize,
                              batch_size=config.batch_size)
    test_iter= data_loader.load_data(texts)
    result = evaluate(test_iter, model, device=config.device, PAD_IDX=data_loader.PAD_IDX)
    return result.numpy()


def evaluate(data_iter, model, device, PAD_IDX):
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            x = x.to(device)
            padding_mask = (x == PAD_IDX).transpose(0, 1)
            logits = model(x, attention_mask=padding_mask)
        model.train()
        return logits.argmax(1)

def read_docx(path):
    # 提取正篇报告
    doc = Document(path)
    list_data = []
    for p in doc.paragraphs:
        list_data.append(p.text)
    data = []
    for i in list_data:
        if (len(i) > 5):
            data.append(i)
    return data

def split_sentences_by_symbols(text_list):
    # 要分割的符号列表
    split_symbols = ["。","；"]
    new=[]
    # 遍历文字列表
    for text in text_list:
        # 逐个符号进行分割
        for symbol in split_symbols:
            # 如果符号在句子中
            if symbol in text:
                # 按照符号分割
                parts = text.split(symbol)
                # 打印分割后的部分
                for part in parts:
                    # print(part.strip())
                    new.append(part.strip())
                # 如果找到一个符号分割，跳过其他符号的分割
                break
        else:
            # 如果没有找到任何分割符号，直接打印原句
            # print(text.strip())
            new.append(text.strip())
    new=[i.replace(" ","") for i in new if len(i)>2]
    return new

def evaluate_exctract(dev_loader, model, mode='dev'):
    # set model to evaluation mode
    model.eval()
    if mode == 'test':
        tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True, skip_special_tokens=True)
    id2label = config.id2label
    pred_tags = [] #预测标签
    sent_data = [] #原始数据
    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            batch_data, batch_token_starts, batch_tags = batch_samples
            if mode == 'test':
                sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                                   if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
            label_masks = batch_tags.gt(-1)  # get padding mask, gt(x): get index greater than x
            # (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]
            # (batch_size, max_len - padding_label_len)
            batch_output = model.crf.decode(batch_output, mask=label_masks)
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
    #输出预测结果
    results=[]
    # print(pred_tags)
    for tag in pred_tags:
        r=get_entities(tag)
        results.append(r)
    return results

def NameEntityExtraction(texts,labels):
    # utils.set_logger(config.log_dir)
    # print(len(data['text']))
    test_dataset = NERDataset(texts, labels, config)
    logging.info("--------Dataset Build!--------")
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    logging.info("--------Get Data-loader!--------")
    # Prepare model
    if config.model_dir is not None:
        model = BertNER.from_pretrained(config.model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to test !--------")
        return
    #预测标签
    result=evaluate_exctract(test_loader, model, mode='test')
    return result

def preprocess(texts):
    data_text=[]
    data_label=[]
    for text in texts:
        text_list=list(text)[:511]
        lens=len(text_list)
        tmp_label=[]
        # data={'text':"",'label':''}
        for i in range(0,lens):
            tmp_label.append('O')
        if(len(tmp_label)==lens):
            data_text.append(text_list)
            data_label.append(tmp_label)
        else:
            print("数据有误，请检查！")
            break
    return data_text,data_label
    
def result_convert(texts,preds,index_dict):
    # 初始化一个空的字典用于存储分组结果
    grouped_pred = {}
    grouped_text= {}
    results_dict = {0: [], 6: [], 7: [], 8: [], 9: []}
    # 遍历 index_dict 进行分组
    for key, indices in index_dict.items():
        # 提取 pred 中相应下标的元素
        grouped_pred[key] = [preds[i] for i in indices]
        grouped_text[key] = [texts[i] for i in indices]
    # 打印分组结果
    for key in grouped_text:
        text_group = grouped_text[key]
        pred_group = grouped_pred[key]
        for text, pred in zip(text_group, pred_group):
            if len(pred)>0:
                for p in pred:
                    words=''
                    w=text[p[1]:p[2] + 1]
                    words=words.join(w)
                    results_dict[key].append({p[0]:words})
    return results_dict

def remove_specific_punctuation(text):
    """
    删除字符串中指定的标点符号，保留汉字和数字。
    这里以删除中英文常见标点为例。
    """
    # 定义需要删除的标点符号集合，包括中文和英文标点
    punctuation_to_remove = r'[.\s·]'
    
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

## 按照系统页面处理文字
def process_text(texts, classified, target_item):
    """
    处理文本：根据目标分类和条件，合并文本并去掉空格。
    
    参数:
    - data: 文本数据列表
    - classified: 分类结果列表
    - target_item: 目标分类标签
    
    返回:
    - 合并并去除空格的文本字符串
    """
    result_text = ''
    for i, item in enumerate(classified):
        if item == target_item and len(texts[i]) > 15:
            if target_item==9:
                result_text += str(texts[i].replace(" ", ''))+str(";\n")
            else:
                result_text += str(texts[i].replace(" ", ''))+str("。")
    return result_text

@Gooey(language='chinese',
       dump_build_config=False,
       program_name=u'Extraction Info From GPR',
       richtext_controls=True,
       encoding="utf-8",
       required_cols=2,
       optional_cols=2,
       default_size=(760, 840),
       menu=[{
           'name': '菜单',
           'items': [{
               'type': 'AboutDialog',
               'menuTitle': '关于',
               'name': 'Extraction Info From GPR',
               'description': 'Copyright © 同济大学',
               'Author': '阿迪力·如苏力',
               'version': 'v1.0.0',
           }]
       }])
def main():
    # 分类名称
    label_map = {0: '报告信息', 1: '项目概况', 2: '任务要求', 3: '规范', 4: '工作原理', 5: '现场布置',
                6: '掌子面信息', 7: '预报结论', 8: '预报成果', 9: '施工建议',10:"其他"}
    
    # Initialize an empty dictionary to store indices for each value
    index_dict = {0: [], 6: [], 7: [], 8: [], 9: []}

    #页面搭建
    parser = GooeyParser(description="信息智能提取demo")
    parser.add_argument('-file_path', widget="FileChooser", default="./xxx.docx", help="文件路径（仅支持docx格式）")
    args = parser.parse_args()
    print(args.file_path)

    print("\n......Classifying Sentences，please wait......\n")
    path= args.file_path
    #文件读取
    if path.endswith("pdf"):
        text = extract_text(path)
        # 调用函数并打印结果
        data = split_into_sentences(text)
    if path.endswith("docx"):
        docx = read_docx(path)
        data = split_sentences_by_symbols(docx)
    #模型推理
    model_config = SentenceClassificationModelConfig()
    classified=SentenceClassification(model_config,data)
    #结果处理
    face_info=[]
    index=0
    for i,item in enumerate(classified):
        # print(data[i],'---',label_map[item])
        if item in [0, 6, 7, 8, 9]:
            index_dict[item].append(index)
            face_info.append(data[i])
            index+=1

    # 调用函数处理不同的分类
    evaluation = process_text(data, classified, 7)
    procaste = process_text(data, classified, 8)
    advise = process_text(data, classified, 9)

    print("\n......Extracting info. please wait......\n")
    texts,labels=preprocess(face_info)
    pred=NameEntityExtraction(texts,labels)

    result=result_convert(texts,pred,index_dict)
    #结果打印
    print("\n...... done ！......\n")

    print(f'-----提取结果:----- \n')
    print("报告文件：",os.path.basename(path))
    # 用于存储已经输出的值
    seen_values = set()
    for key,values in result.items():
        # print(f"{label_map[key]} : \n")
        for item in values:
            v_key = list(item.keys())[0]
            v_value = item[v_key]
            if v_value in seen_values:
                continue  # 如果值已经输出过，跳过此项
            seen_values.add(v_value)  # 将值添加到 seen_values 集合中
            if v_key=="隧道":
                print("隧道:",v_value)

            elif v_key=="线路":
                print("掌子面:",v_value)

            elif v_key=="预报范围":
                print("里程范围:",v_value)

            elif v_key=="长度":
                print("预报范围:",v_value)

            elif v_key=="预报时间":
                print("报告时间:",v_value)

            elif v_key=="预报时间":
                print("报告时间:",v_value)

            elif v_key=="围岩实际等级":
                print("建议围岩等级:",v_value)
            
            elif v_key=="围岩设计等级":
                print("设计围岩等级:",v_value)
    
            # print(f"    {v}  \n")
    print("\n \033[1;92m评估分析:\033[0m")
    print(evaluation,procaste,"\n")
    print("\n \033[1;92m建议措施:\033[0m \n")
    print(advise)

    print("\n...... 结束......\n")

if __name__ == '__main__':
    sys.exit(main())


