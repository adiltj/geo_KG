import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class NERDataset(Dataset):
    def __init__(self, words, labels, config, word_pad_idx=0, label_pad_idx=-1):
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)
        self.label2id = config.label2id
        self.id2label = {_id: _label for _label, _id in list(config.label2id.items())}
        self.dataset = self.preprocess(words, labels)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx
        self.device = config.device

    def preprocess(self, origin_sentences, origin_labels):
        """
        Maps tokens and tags to their indices and stores them in the dict data.
        examples: 
            word:['[CLS]', '浙', '商', '银', '行', '企', '业', '信', '贷', '部']
            sentence:([101, 3851, 1555, 7213, 6121, 821, 689, 928, 6587, 6956],
                        array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]))
            label:[3, 13, 13, 13, 0, 0, 0, 0, 0]
        """
        data = []
        sentences = []
        labels = []
        for line in origin_sentences:
            # replace each token by its index
            # we can not use encode_plus because our sentences are aligned to labels in list type
            words = []
            word_lens = []
            for token in line:
                words.append(self.tokenizer.tokenize(token))
                word_lens.append(len(token))
            # 变成单个字的列表，开头加上[CLS]
            words = ['[CLS]'] + [item for token in words for item in token]
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
            sentences.append((self.tokenizer.convert_tokens_to_ids(words), token_start_idxs))
        for tag in origin_labels:
            label_id = [self.label2id.get(t) for t in tag]
            labels.append(label_id)
        for sentence, label in zip(sentences, labels):
            data.append((sentence, label))
        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def collate_fn(self, batch):
        """
        process batch data, including:
            1. padding: 将每个batch的data padding到同一长度（batch中最长的data长度）
            2. aligning: 找到每个sentence sequence里面有label项，文本与label对齐
            3. tensor：转化为tensor
        """
        sentences = [x[0] for x in batch]
        labels = [x[1] for x in batch]

        # batch length
        batch_len = len(sentences)

        # compute length of longest sentence in batch
        max_len = max([len(s[0]) for s in sentences])
        max_label_len = 0

        # padding data 初始化
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))
        batch_label_starts = []

        # padding and aligning
        for j in range(batch_len):
            cur_len = len(sentences[j][0])
            batch_data[j][:cur_len] = sentences[j][0]
            # 找到有标签的数据的index（[CLS]不算）
            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # padding label
        batch_labels = self.label_pad_idx * np.ones((batch_len, max_label_len))
        for j in range(batch_len):
            cur_tags_len = len(labels[j])
            batch_labels[j][:cur_tags_len] = labels[j]

        # convert data to torch LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_label_starts = torch.tensor(batch_label_starts, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        # shift tensors to GPU if available
        batch_data, batch_label_starts = batch_data.to(self.device), batch_label_starts.to(self.device)
        batch_labels = batch_labels.to(self.device)
        return [batch_data, batch_label_starts, batch_labels]



class Vocab:
    """
    根据本地的vocab文件,构造一个词表
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    print(len(vocab))  # 返回词表长度
    """
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):
    """
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    """
    return Vocab(vocab_path)


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors

class LoadDataset:
    def __init__(self,
                 vocab_path='./vocab.txt',  #
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 split_sep='\n',
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True
                 ):

        """

        :param vocab_path: 本地词表vocab.txt的路径
        :param tokenizer:
        :param batch_size:
        :param max_sen_len: 在对每个batch进行处理时的配置；
                            当max_sen_len = None时，即以每个batch中最长样本长度为标准，对其它进行padding
                            当max_sen_len = 'same'时，以整个数据集中最长样本为标准，对其它进行padding
                            当max_sen_len = 50， 表示以某个固定长度符样本进行padding，多余的截掉；
        :param split_sep: 文本和标签之前的分隔符，默认为'\t'
        :param max_position_embeddings: 指定最大样本长度，超过这个长度的部分将本截取掉
        :param is_sample_shuffle: 是否打乱训练集样本（只针对训练集）
                在后续构造DataLoader时，验证集和测试集均指定为了固定顺序（即不进行打乱），修改程序时请勿进行打乱
                因为当shuffle为True时，每次通过for循环遍历data_iter时样本的顺序都不一样，这会导致在模型预测时
                返回的标签顺序与原始的顺序不一样，不方便处理。

        """
        self.tokenizer = tokenizer
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        # self.UNK_IDX = '[UNK]'

        self.batch_size = batch_size
        self.split_sep = split_sep
        self.max_position_embeddings = max_position_embeddings
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.is_sample_shuffle = is_sample_shuffle

    def data_process(self, texts):
        """
        将每一句话中的每一个词根据字典转换成索引的形式，同时返回所有样本中最长样本的长度
        :param texts: [[一句话],[一句话]]
        :return:
        """
        raw_iter = texts
        data = []
        max_len = 0
        for raw in raw_iter:
            s, l = raw, 0
            tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(s)]
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]  # BERT预训练模型只取前512个字符
            tmp += [self.SEP_IDX]
            tensor_ = torch.tensor(tmp, dtype=torch.long)
            l = torch.tensor(int(l), dtype=torch.long)
            max_len = max(max_len, tensor_.size(0))
            data.append((tensor_, l))
        return data, max_len

    def load_data(self, testx):
        postfix = str(self.max_sen_len)

        test_data, _ = self.data_process(testx)

        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)

        return test_iter

    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            batch_sentence.append(sen)
            batch_label.append(label)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label