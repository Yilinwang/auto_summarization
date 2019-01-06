from torchtext import data, datasets
from pytorch_pretrained_bert import BertTokenizer, BertModel
import random
import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyTokenizer():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def numbericalized_tokenize(self, text):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = torch.zeros(seq_len, device=device) # B x 1 x S

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        
        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
    
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        
        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights
    
    def initHidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)


def train(input_tensor, target_tensor, bert_model, decoder, decoder_optimizer, criterion):
    SOS_token, EOS_token = 2, 3
    decoder_optimizer.zero_grad()
    loss = 0
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    encoded_layers, pooled_output = bert_model(input_tensor)
    encoder_outputs = encoded_layers[-1] #use last layer
    
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_context = torch.zeros(1, decoder.hidden_size, device=device)
    decoder_hidden = decoder.initHidden(input_tensor.size(1))
    
    teacher_forcing_ratio = 0.5
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    
    loss.backward()
    decoder_optimizer.step()
    return loss.item() / target_length


def main():
    tokenizer = MyTokenizer()
    TEXT = data.Field(sequential=True, use_vocab=False, tokenize=tokenizer.numbericalized_tokenize,
             pad_token=0)
    SUMMARY = data.ReversibleField(sequential=True, init_token='<sos>', eos_token='<eos>')
    train_data = data.TabularDataset(path='/home/yilin/summary/data/newsroom/small.json', format='json',
                fields={'text': ('text', TEXT), 'summary': ('summary', SUMMARY)})
    SUMMARY.build_vocab(train_data, max_size=30000)

    hidden_size = 768
    output_size = len(SUMMARY.vocab)
    learning_rate = 0.0001
    n_epochs = 10

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    attn_decoder = AttnDecoderRNN('general', hidden_size, output_size, n_layers=2).to(device)

    decoder_optimizer = optim.Adam(attn_decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(n_epochs):
    running_loss = 0
    for batch in tqdm.tqdm(data.BucketIterator(dataset=train_data, batch_size=4)):
        loss = train(batch.text, batch.summary, bert_model, attn_decoder, decoder_optimizer, criterion)
        running_loss += loss
    
    epoch_loss = running_loss / len(train_data)
    print(f'Epoch: {epoch}, Training Loss: {epoch_loss}')


if __name__ == '__main__':
    main()
