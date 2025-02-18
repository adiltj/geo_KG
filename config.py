import os
import torch
from model import BertConfig

class SentenceClassificationModelConfig:
    def __init__(self):
        self.project_dir = os.path.dirname(__file__)
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'SingleSentenceClassification')
        self.pretrained_model_dir = os.path.join(self.project_dir, "pretrained_Bert")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cpu')
        self.model_save_dir = os.path.join(self.project_dir, 'trained_models','model_for_sentences_classification')
        self.batch_size = 999 #相当于句子列表长度
        self.max_sen_len = None
        self.num_labels = 11
        # 把原始bert中的配置参数也导入进来
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value

print(os.getcwd())

project_dir = os.path.dirname(__file__)
bert_model = os.path.join(project_dir, 'pretrained_Bert')
vocab_path = os.path.join(bert_model, 'vocab.txt')
model_dir = os.path.join(project_dir, 'trained_models','model_for_NER')

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 4
epoch_num = 100
min_epoch_num = 5
patience = 0.0002
patience_num = 100

gpu = ''

if gpu != '':
    device = torch.device(f"cuda:{0}")
else:
    device = torch.device("cpu")

# labels = [
#     "lithology",
#     "rockHardness",
#     "attitude",
#     "weathering",
#     "joints",
#     "jointCondition",
#     "waterCondition",
#     "waterSeepage",
#     "rockIntegrity",
#     "rockStability",
#     "designRockGrade",
#     "actualRockGrade",
#     "fault"
# ]

# label2id = {
#     "O": 0,
#     "B-lithology":1,
#     "B-rockHardness":2,
#     "B-attitude":3,
#     "B-weathering":4,
#     "B-joints":5,
#     "B-jointCondition":6,
#     "B-waterCondition":7,
#     "B-waterSeepage":8,
#     "B-rockIntegrity":9,
#     "B-rockStability":10,
#     "B-designRockGrade":11,
#     "B-actualRockGrade":12,
#     "B-fault":13,
#     "I-lithology":14,
#     "I-rockHardness":15,
#     "I-attitude":16,
#     "I-weathering":17,
#     "I-joints":18,
#     "I-jointCondition":19,
#     "I-waterCondition":20,
#     "I-waterSeepage":21,
#     "I-rockIntegrity":22,
#     "I-rockStability":23,
#     "I-designRockGrade":24,
#     "I-actualRockGrade":25,
#     "I-fault":26,
# }

labels =[
    "隧道",
    "线路",
    "预报范围",
    "预报时间",
    "里程桩号",
    "地层岩性",
    "地层颜色",
    "岩层产状",
    "节理产状",
    "稳定性",
    "坚硬程度",
    "风化程度",
    "结构类型",
    "节理发育程度",
    "完整性",
    "地下水情况",
    "地质构造影响",
    "围岩设计等级",
    "围岩实际等级",
    "里程范围",
    "出水情况",
    "长度",
    "反射振幅变化",
    "反射频率变化",
    "同相轴连续性",
    "开挖措施",
    "支护措施",
    "监测措施",
    "排水措施",
    "标段",]

label2id = {
    "O":0,
    "B-隧道":1,
    "B-线路":2,
    "B-预报范围":3,
    "B-预报时间":4,
    "B-里程桩号":5,
    "B-地层岩性":6,
    "B-地层颜色":7,
    "B-岩层产状":8,
    "B-节理产状":9,
    "B-稳定性":10,
    "B-坚硬程度":11,
    "B-风化程度":12,
    "B-结构类型":13,
    "B-节理发育程度":14,
    "B-完整性":15,
    "B-地下水情况":16,
    "B-地质构造影响":17,
    "B-围岩设计等级":18,
    "B-围岩实际等级":19,
    "B-里程范围":20,
    "B-出水情况":21,
    "B-长度":22,
    "B-反射振幅变化":23,
    "B-反射频率变化":24,
    "B-同相轴连续性":25,
    "B-开挖措施":26,
    "B-支护措施":27,
    "B-监测措施":28,
    "B-排水措施":29,
    "B-标段":30,
    "I-隧道":31,
    "I-线路":32,
    "I-预报范围":33,
    "I-预报时间":34,
    "I-里程桩号":35,
    "I-地层岩性":36,
    "I-地层颜色":37,
    "I-岩层产状":38,
    "I-节理产状":39,
    "I-稳定性":40,
    "I-坚硬程度":41,
    "I-风化程度":42,
    "I-结构类型":43,
    "I-节理发育程度":44,
    "I-完整性":45,
    "I-地下水情况":46,
    "I-地质构造影响":47,
    "I-围岩设计等级":48,
    "I-围岩实际等级":49,
    "I-里程范围":50,
    "I-出水情况":51,
    "I-长度":52,
    "I-反射振幅变化":53,
    "I-反射频率变化":54,
    "I-同相轴连续性":55,
    "I-开挖措施":56,
    "I-支护措施":57,
    "I-监测措施":58,
    "I-排水措施":59,
    "I-标段":60,
}

id2label = {_id: _label for _label, _id in list(label2id.items())}
