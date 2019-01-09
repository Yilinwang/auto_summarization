from pytorch_pretrained_bert import BertTokenizer, BertModel
from newsroom import jsonl
import torch
import json
import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
     print(torch.cuda.get_device_name(device))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path')
    parser.add_argument('-t', '--target_path')
    args = parser.parse_args()
    return args


def main(args):
    args.data_path = '/home/yilin10945/summary/data/newsroom/train.data'
    args.target_path = './data/train_features.json'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    bert_model = bert_model.to(device)
    bert_model_cpu = BertModel.from_pretrained('bert-base-uncased')
    bert_model_cpu.eval()

    train_data = jsonl.open(args.data_path, gzip=True).read()
    with open(args.target_path, 'w') as fp:
        for idx, doc in enumerate(tqdm.tqdm(train_data)):
            if doc['text']:
                doc_features = list()
                for sen in doc['text'].split('\n\n'):
                    if len(sen) > 1:
                        input_tok = tokenizer.tokenize(sen[:512])
                        input_list = tokenizer.convert_tokens_to_ids(input_tok)
                        try:
                            input_tensor = torch.tensor(input_list).view(1, -1).to(device)
                            encoded_layers, pooled_output = bert_model(input_tensor)
                        except:
                            input_tensor = input_tensor.to('cpu')
                            encoded_layers, pooled_output = bert_model_cpu(input_tensor)
                        features = encoded_layers[-1].view(-1, 768).tolist()
                        doc_features.extend(features)
                doc['id'] = idx
                doc['bert_features'] = doc_features
                print(json.dumps(doc), file=fp)


if __name__ == '__main__':
    main(parse_args())
