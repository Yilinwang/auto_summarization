from pytorch_pretrained_bert import BertTokenizer, BertModel
from newsroom import jsonl
import torch
import json
import tqdm


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path')
    parser.add_argument('-t', '--target_path')
    args = parser.parse_args()
    return args


def main(args):
    args.data_path = '../data/newsroom/train.data'
    args.target_path = './data/train_features.json'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()

    train_data = jsonl.open(args.data_path, gzip=True).read()
    with open(args.target_path, 'w') as fp:
        for doc in tqdm.tqdm(train_data):
            if doc['text']:
                input_list = tokenizer.convert_tokens_to_ids(
                               tokenizer.tokenize(doc['text']))
                input_tensor = torch.tensor(input_list).view(-1, 1)
                encoded_layers, pooled_output = bert_model(input_tensor)
                features = encoded_layers[-1].view(-1, 768).tolist()
                doc['bert_features'] = features
                print(json.dumps(doc), file=fp)


if __name__ == '__main__':
    main(parse_args())
