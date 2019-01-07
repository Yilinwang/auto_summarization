from torchtext import data, datasets
from pytorch_pretrained_bert import BertTokenizer, BertModel
import random
import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
 print(torch.cuda.get_device_name(device))


class MyTokenizer():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def numbericalized_tokenize(self, text):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))


class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


def train(input_tensor, target_tensor, bert_model, decoder, decoder_optimizer, criterion):
    SOS_token, EOS_token = 2, 3
    decoder_optimizer.zero_grad()
    loss = 0
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    batch_size = input_tensor.size(1)
    
    encoded_layers, pooled_output = bert_model(input_tensor)
    encoder_outputs = encoded_layers[-1] #use last layer
    encoder_outputs = encoder_outputs.to(device)
    
    decoder_input = torch.tensor([[SOS_token for _ in range(batch_size)]], device=device)
    decoder_hidden = None
    
    teacher_forcing_ratio = 0.5
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di].view(1, -1)
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = torch.tensor([[topi[i][0] for i in range(batch_size)]], device=device)

            loss += criterion(decoder_output, target_tensor[di])
    
    loss.backward()
    decoder_optimizer.step()
    return loss.item() / target_length


def main():
    tokenizer = MyTokenizer()

    TEXT = data.Field(sequential=True, use_vocab=False, tokenize=tokenizer.numbericalized_tokenize,
             pad_token=0)
    SUMMARY = data.ReversibleField(sequential=True, init_token='<sos>', eos_token='<eos>')

    print('Data Loading...')
    train_data = data.TabularDataset(path='/home/yilin10945/summary/data/newsroom/train.200.json', format='json',
                fields={'text': ('text', TEXT), 'summary': ('summary', SUMMARY)})
    SUMMARY.build_vocab(train_data, max_size=30000)
    #import pickle
    #pickle.dump((train_data, TEXT, SUMMARY), open('model/processed_data.pkl', 'wb'))
    print('Data Loaded!!!')

    hidden_size = 768
    vocab_size = len(SUMMARY.vocab)
    learning_rate = 0.0001
    n_epochs = 10
    batch_size = 16

    embedding = nn.Embedding(vocab_size, hidden_size)
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    attn_decoder = LuongAttnDecoderRNN('general', embedding, hidden_size, vocab_size, 1, 0.1).to(device)

    decoder_optimizer = optim.Adam(attn_decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    print('Start Training...')
    for epoch in range(n_epochs):
        running_loss = 0
        step = 0
        for batch in tqdm.tqdm(data.BucketIterator(dataset=train_data, batch_size=batch_size)):
            loss = train(batch.text, batch.summary.to(device), bert_model, attn_decoder, decoder_optimizer, criterion)
            running_loss += loss
            step += batch_size

            if step % 128 == 0:
                print(f'Step: {step}, Training Loss: {running_loss/step}')
                torch.save(attn_decoder.state_dict(), f'model/{step}.pt')
        
        epoch_loss = running_loss / len(train_data)
        print(f'Epoch: {epoch}, Training Loss: {epoch_loss}')


if __name__ == '__main__':
    main()
