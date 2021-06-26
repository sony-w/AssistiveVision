__author__ = 'sony-w'
__version__ = '1.0'

import torch
import torch.nn as nn
import torchvision.models as models

from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from .fn import embedding_layer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    
    def __init__(self, embedding_size):
        """
        Load pre-trained resnext101_32x8d and replace top fully-connected layer
        
        Parameters:
            embedding_size(int): embedding size for tokens
        """
        
        super(Encoder, self).__init__()
        resnext = models.resnext101_32x8d(pretrained=True) #torch.hub.load('pytorch/vision:v0.9.0', 'resnext101_32x8d', pretrained=True)
        # remove the last fully connected layer
        modules = list(resnext.children())[:-1]
        
        self.resnext = nn.Sequential(*modules)
        self.embed = nn.Sequential(
            nn.Linear(resnext.fc.in_features, embedding_size),
            nn.Dropout(p=0.5)
        )
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    
    
    def forward(self, images):
        """
        Extract feature vectors from input images
        """
        
        with torch.no_grad():
            features = self.resnext(images)
        
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)

        return features
    
    
class Decoder(nn.Module):
    
    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers=1, 
                 embedding_matrix=None, train_embedding=True):
        """
        Define hyper-parameters and build the layers
        """
        
        super(Decoder, self).__init__()
        self.embed =  embedding_layer(num_embeddings=vocab_size, embedding_dim=embedding_size,        # nn.Embedding(vocab_size, embedding_size)
                                      embedding_matrix=embedding_matrix, trainable=train_embedding) 
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions, lengths):
        """
        Decode image features and generate captions
        """
        
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        
        return outputs
    
    def sample(self, features, states=None, max_len=40):
        """
        Samples captions in batch for given pre-processed image tensor with greedy search
        """

        inputs = features.unsqueeze(1)
        sampled_ids = []

        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted).unsqueeze(1)

        sampled_ids = torch.stack(sampled_ids, 1)

        return sampled_ids
        
    
    def sample_beam_search(self, features, states=None, max_len=40, beam_width=5):
        """
        Samples captions in batch for given pre-proccessed image tensor returning top n predicted sentences
        """

        inputs = features.unsqueeze(1)
        idx_sequences = [[[], 0.0, inputs, states]]

        for _ in range(max_len):
            # placeholder for all candidates at each step
            candidates = []
            # predict the next word idx for each of the top sequences
            for idx_seq in idx_sequences:
                hiddens, states = self.lstm(idx_seq[2], idx_seq[3])
                outputs = self.linear(hiddens.squeeze(1))
                # transform outputs to log probs to prevent floating-point underflow
                # caused by multiplying very small probabilities
                log_probs = F.log_softmax(outputs, -1)
                top_log_probs, top_idx = log_probs.topk(beam_width, 1)
                top_idx = top_idx.squeeze(0)
                # create a new set of top sentences for next round
                for i in range(beam_width):
                    next_idx_seq, log_prob = idx_seq[0][:], idx_seq[1]
                    next_idx_seq.append(top_idx[i].item())
                    log_prob += top_log_probs[0][i].item()
                    inputs = self.embed(top_idx[i].unsqueeze(0)).unsqueeze(0)
                    candidates.append([next_idx_seq, log_prob, inputs, states])

            # keep only the top sequences according to the total log probability
            ordered = sorted(candidates, key=lambda x: x[1], reverse=True)
            idx_sequences = ordered[:beam_width]

        return [idx_seq[0] for idx_seq in idx_sequences]


    
class EncoderAttention(nn.Module):
    
    def __init__(self, encoded_image_size=14, fine_tune=False):
        
        super(EncoderAttention, self).__init__()
        self.encoded_image_size = encoded_image_size
        resnext = models.resnext101_32x8d(pretrained=True)
        # remove the linear and pooling layer
        modules = list(resnext.children())[:-2]
        
        self.resnext = nn.Sequential(*modules)
        # resize the image to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune(fine_tune=fine_tune)
    
    def forward(self, images):
        
        outputs = self.resnext(images)
        outputs = self.adaptive_pool(outputs)
        outputs = outputs.permute(0, 2, 3, 1)
        
        return outputs
    
    
    def fine_tune(self, fine_tune=False):
        """
        Allow or prevent the gradients computation for convolutional blocks
        """
        
        for p in self.resnext.parameters():
            p.requires_grad = False
        
        # only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnext.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
        

class Attention(nn.Module):
    
    def __init__(self, encoder_dim=2048, decoder_dim=512, attention_dim=512):
        
        super(Attention, self).__init__()
        # linear layer to transform encoded image
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # linear layer to transform decoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # linear layer to compute values to be softmax-ed
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    
    def forward(self, encoder_outputs, decoder_hiddens):
        attention_1 = self.encoder_att(encoder_outputs)
        attention_2 = self.decoder_att(decoder_hiddens)
        attention = self.full_att(self.relu(attention_1 + attention_2.unsqueeze(1))).squeeze(2)
        
        alpha = self.softmax(attention)
        attention_weighted_encoding = (encoder_outputs * alpha.unsqueeze(2)).sum(dim=1)
        
        return attention_weighted_encoding, alpha
    

class DecoderAttention(nn.Module):
    
    def __init__(self, attention_dim, embedding_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5,
                 alpha_c=1, embedding_matrix=None, train_embedding=True):
        
        super(DecoderAttention, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.alpha_c = alpha_c
        
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = embedding_layer(num_embeddings=vocab_size, embedding_dim=embedding_dim,        # nn.Embedding(vocab_size, embedding_dim)
                                         embedding_matrix=embedding_matrix, trainable=train_embedding)
        
        # linear layers to determine initial states of LSTMs 
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        # gating scalars and sigmoid layer
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        
        # LSTM
        self.decode_step = nn.LSTMCell(embedding_dim + encoder_dim, decoder_dim, bias=True)
        # dropout layer
        self.dropout = nn.Dropout(p=dropout)
        # linear layer to find scores over vocabulary
        self.fc = nn.Linear(decoder_dim, vocab_size)
        
        #self.criterion = nn.CrossEntropyLoss()
        
        #self.linear_o = nn.Linear(embed_size, self.vocab_size)
        #self.linear_h = nn.Linear(decoder_dim, embed_size)
        #self.linear_z = nn.Linear(encoder_dim, embed_size)
        
        self.init_weights()
    
    
    def init_weights(self):
        """
        Initialize parameters with values from the uniform distribution for easier convergence
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    
    
    def init_hidden_states(self, encoder_outputs):
        """
        Create initial hidden and cell states for decoder's LSTM based on the encoded images
        """
        
        mean_encoder_outputs = encoder_outputs.mean(dim=1)
        h = self.init_h(mean_encoder_outputs)
        c = self.init_c(mean_encoder_outputs)
        
        return h, c
    
    def forward(self, encoder_outputs, encoded_captions, caption_len):
        
        batch_size = encoder_outputs.size(0)
        encoder_dim = encoder_outputs.size(-1)

        # flatten image
        encoder_outputs = encoder_outputs.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_outputs.size(1)
    
        # sort input data (encoder_outputs) by decreasing lengths
        caption_len, sorted_idx = caption_len.squeeze(1).sort(dim=0, descending=True)
        encoder_outputs = encoder_outputs[sorted_idx]
        encoded_captions = encoded_captions[sorted_idx]
            
        embeddings = self.embedding(encoded_captions).type(torch.FloatTensor).to(device)
        h, c = self.init_hidden_states(encoder_outputs)
        decode_len = (caption_len - 1).tolist()
        
        # placehoder tensors to keep word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_len), self.vocab_size).to(encoder_outputs.device)
        alphas = torch.zeros(batch_size, max(decode_len), num_pixels).to(encoder_outputs.device)
        
        # For each time-step, decode by attention-weighing the encoder's output
        # based on the previous decoder's hidden state output
        # then predict a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_len)):
            batch_size_t = sum([l > t for l in decode_len])
            attention_weighted_encoding, alpha = self.attention(encoder_outputs[:batch_size_t], h[:batch_size_t])
            
            # gating scalar
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )
            
            #h_embedded = self.linear_h(h)
            #attention_embedded = self.linear_z(attention)
            #preds = self.linear_o(self.dropout(embeddings[:batch_size_t, t, :] + h_embedded + attention_embedded))
            
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        
        return predictions, encoded_captions, decode_len, alphas, sorted_idx
    
    def loss(self, outputs, targets, alphas):
        
        loss = self.criterion(outputs, targets.cpu())
        loss += self.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        
        return loss
    
    
    def sample(self, features, startseq_idx, states=None, max_len=40, return_alpha=False):

        batch_size = features.size(0)
        encoder_dim = features.size(3) #features.size(-1)
        encoder_img_size = features.size(1)
        
        features = features.view(batch_size, -1, encoder_dim)
        
        h, c = self.init_hidden_states(features)

        sampled_ids, alphas = [], []        

        prev_word = torch.LongTensor([[startseq_idx]] * batch_size).to(features.device) 
        # torch.LongTensor([[self.vocab.word2idx['<start>']]]).to(device)

        for t in range(max_len):
            embeddings = self.embedding(prev_word).squeeze(1)
            attention, alpha = self.attention(features, h)
            alpha = alpha.view(-1, encoder_img_size, encoder_img_size).unsqueeze(1)
            
            gate = self.sigmoid(self.f_beta(h))
            
            attention = gate * attention
            h, c = self.decode_step(torch.cat([embeddings, attention], dim=1), (h, c))
            
            #h_embedded = self.linear_h(h)
            #attention_embedded = self.linear_z(h)
            #preds = self.linear_o(self.dropout(embeddings + h_embedded + attention_embedded))
            
            preds = self.fc(h)
            predicted = preds.argmax(1) #torch.max(preds, dim=1)
            
            prev_word = predicted.unsqueeze(1)
            sampled_ids.append(predicted)
            alphas.append(alpha)
        
        sampled_ids = torch.stack(sampled_ids, 1)
        
        return (sampled_ids, torch.cat(alphas, 1)) if return_alpha else sampled_ids
    
    
class Transformer(nn.Module):
    
    def __init__(self, embedding_size, hidden_size, vocab_size, num_layers=1, 
                 embedding_matrix=None, train_embedding=True):
        
        super().__init__()
        self.encoder = Encoder(embedding_size)
        self.decoder = Decoder(embedding_size, hidden_size, vocab_size, num_layers, 
                               embedding_matrix=embedding_matrix, train_embedding=train_embedding)
    
    
    def forward(self, images, captions, length):
        
        features = self.encoder(images)
        outputs = self.decoder(features, captions, length)
        
        return outputs
    

    def sample(self, images, max_len=40):
        
        features = self.encoder(images)
        captions = self.decoder.sample(features=features, max_len=max_len)
        
        return captions
    
    
    def sample_beam_search(self, images, max_len=40, beam_width=5):
        features = self.encoder(images)
        captions = self.decoder.sample_beam_search(features=features, max_len=max_len, beam_width=beam_width)
        
        return captions


    
class TransformerAttention(nn.Module):
    
    def __init__(self, encoded_image_size, attention_dim, embedding_dim, decoder_dim, vocab_size, encoder_dim=2048, 
                 dropout=0.5, alpha_c=1.0, embedding_matrix=None, train_embedding=True, fine_tune=False):
        
        super().__init__()
        self.encoder = EncoderAttention(encoded_image_size=encoded_image_size, fine_tune=fine_tune)
        self.decoder = DecoderAttention(attention_dim, embedding_dim, decoder_dim, vocab_size, encoder_dim=encoder_dim, dropout=dropout,
                                       alpha_c=alpha_c, embedding_matrix=embedding_matrix, train_embedding=train_embedding)
        

        
    def forward(self, images, encoded_captions, caption_len):
        
        encoder_outputs = self.encoder(images)
        decoder_outputs = self.decoder(encoder_outputs, encoded_captions, caption_len.unsqueeze(1))
        
        return decoder_outputs
    
    
    def sample(self, images, startseq_idx, max_len=40, return_alpha=False):
        
        encoder_outputs = self.encoder(images)        
        return self.decoder.sample(encoder_outputs, startseq_idx, max_len=max_len, return_alpha=return_alpha)
    
    