import argparse
import os
import string
import torch
import logging
import json
import gc

from collections import defaultdict

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm.auto import tqdm
from pathlib import Path

from loader.dataset import VizwizDataset
from loader.model import ModelS3
from commons.utils import embedding_matrix, tensor_to_word_fn
from models.resnext101 import TransformerAttention
from eval.metrics import bleu, cider, rouge, spice, meteor, bleu_score_fn

from IPython.core.display import HTML

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# change this flag whether sagemaker is being used or not
is_sagemaker = True

# log setup
log_path = 'logs'
log_file = 'vision.logs'

if is_sagemaker:
    log_path = '/opt/ml/output/failure'

# create log dir if it does not exist
os.makedirs(log_path, exist_ok=True)
    
logger = logging.getLogger(__name__)
# log handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(os.path.join(log_path, log_file), mode='a')
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)
# log format
c_format = logging.Formatter(fmt='%(name)s :: %(asctime)s :: %(levelname)s - %(message)s', datefmt='%b-%d-%y %H:%M:%S')
f_format = logging.Formatter(fmt='%(name)s :: %(asctime)s :: %(levelname)s - %(message)s', datefmt='%b-%d-%y %H:%M:%S')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

logger.addHandler(c_handler)
logger.addHandler(f_handler)


def fit(dataloaders, model, loss_fn, optimizer, batch_size, desc=''):
    
    LOG_INTERVAL = 25 * (256 // batch_size)
    means = dict()
    
    for phase in ['train', 'val']:
        acc = 0.0
        loss = 0.0
        
        if phase == 'train':
            model.train()
        else:
            model.eval()
            
        t = tqdm(iter(dataloaders[phase]), desc=f'{desc} ::: {phase}')
        for batch_idx, batch in enumerate(t):
            images, captions, lengths, fname, image_id = batch
    
            if phase == 'train':
                optimizer.zero_grad()
            
            scores, captions_sorted, decode_len, alphas, sort_idx = model(images, captions, lengths)
            # exclude <start> and only includes after <start> to <end>
            targets = captions_sorted[:, 1:]
            # remove pads or timesteps that were not decoded
            scores = pack_padded_sequence(scores, decode_len, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_len, batch_first=True)[0]

            loss = loss_fn(scores, targets)
    
            if phase == 'train':
                loss.backward()
                optimizer.step()

            acc += (torch.argmax(scores, dim=1) == targets).sum().float().item() / targets.size(0)
            loss += loss.item()

            t.set_postfix({
                'loss': loss / (batch_idx + 1),
                'acc': acc / (batch_idx + 1)
            }, refresh=True)

            if (batch_idx + 1) % LOG_INTERVAL == 0 :
                print(f'{desc}_{phase} {batch_idx + 1}/{len(dataloaders[phase])} '
                      f'{phase}_loss: {loss / (batch_idx + 1):.4f} '
                      f'{phase}_acc: {acc / (batch_idx + 1):.4f}')
                
            # release gpu memory
            del images
            del captions
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        means[phase] = loss / len(dataloaders[phase])
    
    return means['train'], means['val']
    

def detokenize(tokens):
    return ''.join([' ' + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


def evaluate(data_loader, model, bleu_score_fn, tensor_to_word_fn, vocabulary, desc=''):
    
    model.eval()
    
    pred_byfname = dict()
    caps_byfname = defaultdict(list)
    scores = dict()
    
    running_bleu = [0.0] * 5
    
    t = tqdm(iter(data_loader), desc=f'{desc}')
    for batch_idx, batch in enumerate(t):
        images, captions, lengths, fname, image_id = batch
        outputs = tensor_to_word_fn(model.sample(images, startseq_idx=vocabulary.word2idx['<start>']).cpu().numpy())
        
        for i in range(1, 5):
            running_bleu[i] += bleu_score_fn(reference_corpus=captions, candidate_corpus=outputs, n=i)
        t.set_postfix({
            'bleu1': running_bleu[1] / (batch_idx + 1),
            'bleu4': running_bleu[4] / (batch_idx + 1)
        }, refresh=True)
        
        for f, o, c in zip(fname, outputs, captions):
            if not f in pred_byfname:
                pred_byfname[f] = [detokenize(o)]
            caps_byfname[f].append(detokenize(c))
        
        # release gpu memory
        del images
        del captions
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # mean running_bleu score
    for i in range(1, 5):
        running_bleu[i] /= len(data_loader)
    scores['bleu'] = running_bleu

    # calculate overall score
    scores['coco_bleu'] = bleu(caps_byfname, pred_byfname, verbose=0)
    scores['cider'] = cider(caps_byfname, pred_byfname)
    scores['rouge'] = rouge(caps_byfname, pred_byfname)
    scores['spice'] = spice(caps_byfname, pred_byfname)
    scores['meteor'] = meteor(caps_byfname, pred_byfname)
    
    return scores


def generate_captions(dataloader, model, tensor_to_word_fn, vocabulary, desc=''):
    rlist = []
    
    t = tqdm(iter(dataloader), desc=f'{desc}')
    for batch_idx, batch in enumerate(t):
        images, fname, image_id = batch
        outputs = tensor_to_word_fn(model.sample(images, startseq_idx=vocabulary.word2idx['<start>']).cpu().numpy())

        for out, img in zip(outputs, image_id):
            result = dict(
                image_id = int(img),
                caption = detokenize(out)
            )
            rlist.append(result)

        # release gpu memory
        del images
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    results = dict(
        results = rlist
    )
    
    return results


def main(args):

    BUCKET = args.bucket
    MODEL = args.model_name
    GLOVE_DIR = args.glove_dir

    ENCODER_DIM = args.encoder_dim
    EMBEDDING_DIM = args.embedding_dim
    ATTENTION_DIM = args.attention_dim
    DECODER_DIM = args.decoder_dim

    TRAIN_EMBEDDING = args.train_embedding
    FINE_TUNE = args.fine_tune

    BATCH_SIZE = args.batch_size
    DROPOUT_RATE = args.dropout_rate
    ALPHA_C = args.alpha_c
    LR = args.lr
    NUM_EPOCHS = args.num_epochs

    LOCAL_PATH = args.local_path
    if is_sagemaker:
        LOCAL_PATH = '/opt/ml/model'
    
    KEY_PATH = args.key_path
    
    CAPTIONS_PATH = args.captions_path
    if is_sagemaker:
        CAPTIONS_PATH = '/opt/ml/output'
    
    VERSION = args.version

    MODEL_NAME = f'{MODEL}_b{BATCH_SIZE}_emb{EMBEDDING_DIM}'
    # load vizwiz train dataset
    print('>> loading train dataset... <<')
    train = VizwizDataset(dtype='train', ret_type='tensor', copy_img_to_mem=False, device=device, #partial=100, 
                          is_sagemaker=is_sagemaker, logger=logger)
    vocabulary = train.getVocab()
    # load vizwiz val and eval dataset
    print('\n>> loading val eval dataset... <<')
    val = VizwizDataset(dtype='val', ret_type='tensor', copy_img_to_mem=False, vocabulary=vocabulary, device=device, #partial=50, 
                       is_sagemaker=is_sagemaker, logger=logger)
    print('\n>> loading eval dataset... <<')
    val_eval = VizwizDataset(dtype='val', ret_type='corpus', copy_img_to_mem=False, vocabulary=vocabulary, device=device, #partial=50, 
                            is_sagemaker=is_sagemaker, logger=logger)
    # load vizwiz test dataset
    print('\n>> loading test dataset... <<')
    test = VizwizDataset(dtype='test', ret_type='corpus', copy_img_to_mem=False, vocabulary=vocabulary, device=device, #partial=10, 
                        is_sagemaker=is_sagemaker, logger=logger)
    
    print('\n>> generating word embedding matrix... <<')
    embedding_mtx = embedding_matrix(embedding_dim=EMBEDDING_DIM, word2idx=vocabulary.word2idx, glove_dir=GLOVE_DIR)
    print('done!!')
    
    vocab_size = len(vocabulary.vocab)
    transformer = TransformerAttention(encoded_image_size=14, attention_dim=ATTENTION_DIM, embedding_dim=EMBEDDING_DIM, 
                                       decoder_dim=DECODER_DIM, vocab_size=vocab_size, encoder_dim=ENCODER_DIM, dropout=DROPOUT_RATE,
                                       alpha_c=ALPHA_C, embedding_matrix=embedding_mtx, train_embedding=TRAIN_EMBEDDING, 
                                       fine_tune=FINE_TUNE).to(device) 

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=train.pad_value).to(device)
    corpus_bleu_score_fn = bleu_score_fn(4, 'corpus')
    tensor2word_fn = tensor_to_word_fn(idx2word=vocabulary.idx2word)

    params = transformer.parameters()
    optimizer = torch.optim.Adam(params=params, lr=LR)
    
    train_transformations = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.RandomCrop(256),  # get 256x256 crop from random location
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))
    ])

    eval_transformations = transforms.Compose([
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.CenterCrop(256),  # get 256x256 crop from random location
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                             (0.229, 0.224, 0.225))
    ])

    train.transformations = train_transformations
    val.transformations = train_transformations
    val_eval.transformations = eval_transformations
    test.transformations = eval_transformations
    
    dataloaders = dict(
        train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, sampler=None, pin_memory=False),
        val = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, sampler=None, pin_memory=False)
    )

    eval_collate_fn = lambda batch: (torch.stack([x[0] for x in batch]), [x[1] for x in batch], [x[2] for x in batch], 
                                     [x[3] for x in batch], [x[4] for x in batch])
    test_collate_fn = lambda batch: (torch.stack([x[0] for x in batch]), [x[1] for x in batch], [x[2] for x in batch])

    val_eval_loader = DataLoader(val_eval, batch_size=BATCH_SIZE, shuffle=False, sampler=None, pin_memory=False, collate_fn=eval_collate_fn)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, sampler=None, pin_memory=False, collate_fn=test_collate_fn)
    
    print('\n>> training and evaluating model... <<')
    train_loss_min = 100
    val_loss_min = 100
    val_bleu4_max = 0.0

    model_bin = ModelS3(is_sagemaker=is_sagemaker, logger=logger)
    transformer_best = None

    for epoch in range(NUM_EPOCHS):
        train_loss, val_loss = fit(dataloaders, model=transformer, loss_fn=loss_fn, optimizer=optimizer, batch_size=BATCH_SIZE,
                                  desc=f'Epoch {epoch+1} of {NUM_EPOCHS}')

        with torch.no_grad():
            scores =  evaluate(val_eval_loader, model=transformer, bleu_score_fn=corpus_bleu_score_fn, 
                               tensor_to_word_fn=tensor2word_fn, vocabulary=vocabulary, desc='Eval Score')

            print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
            print('=' * 95)
            print(''.join([f'val_bleu{i}: {scores["bleu"][i]:.4f} ' for i in range(1, 5)]))
            print(''.join([f'val_coco_bleu{i + 1}{":":>5} {scores["coco_bleu"][0][i]:.4f} ' for i in range(0, 4)]))
            print(f'val_cider{":":>5} {scores["cider"][0]:.4f}')
            print(f'val_rouge{":":>5} {scores["rouge"][0]:.4f}')
            print(f'val_spice{":":>5} {scores["spice"][0]:.4f}')
            print(f'val_meteor{":":>5} {scores["meteor"][0]:.4f}')
            print('-' * 95)

            state = dict(
                epoch = epoch + 1,
                state_dict = transformer.state_dict(),
                train_loss_latest = train_loss,
                val_loss_latest = val_loss,
                train_loss_min = min(train_loss, train_loss_min),
                val_loss_min = min(val_loss, val_loss_min),
                val_bleu1 = scores['bleu'][1],
                val_bleu4 = scores['bleu'][4],
                val_bleu4_max = max(scores['bleu'][4], val_bleu4_max),
                val_coco_bleu1 = scores['coco_bleu'][0][0],
                val_coco_bleu4 = scores['coco_bleu'][0][3],
                val_cider = scores['cider'][0],
                val_rouge = scores['rouge'][0], 
                val_spice = scores['spice'][0],
                val_meteor = scores['meteor'][0]
            )

            if scores['bleu'][4] > val_bleu4_max:
                val_bleu4_max = scores['bleu'][4]
                fname = f'{MODEL_NAME}_best_v{VERSION}.pt'
                # keep the best transformer
                transformer_best = transformer
                model_bin.save(state, os.path.join(LOCAL_PATH, fname), os.path.join(KEY_PATH, fname))
    
            # save as checkpoint
            fname = f'{MODEL_NAME}_ep{epoch + 1}_chkpoint_v{VERSION}.pt'
            model_bin.save(state, os.path.join(LOCAL_PATH, fname), os.path.join(KEY_PATH, fname))

    print('done!!')
    
    print('\n>> saving model... <<')
    fname = f'{MODEL_NAME}_ep{NUM_EPOCHS}_latest_v{VERSION}.pt'
    model_bin.save(state, os.path.join(LOCAL_PATH, fname), os.path.join(KEY_PATH, fname))
    print('done!!')
    
    print('\n>> generating and saving test captions... <<')
    results = generate_captions(test_loader, model=transformer, tensor_to_word_fn=tensor2word_fn, 
                                vocabulary=vocabulary, desc='captioning ::: test')
    fname = f'{MODEL_NAME}_ep{NUM_EPOCHS}_latest_v{VERSION}.json'
    model_bin.save_captions(results, os.path.join(CAPTIONS_PATH, fname))
    print('done!!')
    
    
if __name__ == '__main__':
    
    hyperparams = dict()
    if is_sagemaker:
        prefix     = '/opt/ml/'
        param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

        with open(param_path, 'r') as params:
            hyperparams = json.load(params)
    
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--bucket', type=str, default=hyperparams.get('bucket', 'assistive-vision'), help='main S3 bucket name as default storage')
    parser.add_argument('--model_name', type=str, default=hyperparams.get('model_name', 'resnext101_attention'), help='model name identifier')
    parser.add_argument('--glove_dir', type=str, default='annotations/glove', help='directory to store glove embedding')
    
    parser.add_argument('--encoder_dim', type=int, default=int(hyperparams.get('encoder_dim', 2048)), help='dimension of image embedding in attention network')
    parser.add_argument('--embedding_dim', type=int, default=int(hyperparams.get('embedding_dim', 300)), help='dimension of word embedding vectors')
    parser.add_argument('--attention_dim', type=int, default=int(hyperparams.get('attention_dim', 256)), help='dimension of attention network')
    parser.add_argument('--decoder_dim', type=int, default=int(hyperparams.get('decoder_dim', 256)), help='dimension of decoder network')
    
    parser.add_argument('--train_embedding', type=bool, default=bool(hyperparams.get('train_embedding', True)), help='flag to re-train the word embedding layer')
    parser.add_argument('--fine_tune', type=bool, default=bool(hyperparams.get('fine_tune', False)), help='flag to re-train the lower layers of feature extraction')
    
    parser.add_argument('--batch_size', type=int, default=int(hyperparams.get('batch_size', 128)), help='size of batch for each train and validation epochs')
    parser.add_argument('--dropout_rate', type=float, default=float(hyperparams.get('dropout_rate', 0.5)), help='dropout ratio for train regularization')
    parser.add_argument('--alpha_c', type=float, default=float(hyperparams.get('alpha_c', 1.0)), help='weight assigned to the second loss function')
    parser.add_argument('--lr', type=float, default=float(hyperparams.get('lr', 5.e-4)), help='learning rate for adam optimizer')
    parser.add_argument('--num_epochs', type=int, default=int(hyperparams.get('num_epochs', 10)), help='number of train and validation epochs')
    
    parser.add_argument('--local_path', type=str, default='bin/', help='local path location for model repo')
    parser.add_argument('--key_path', type=str, default='bin/', help='s3 path location for model repo')
    parser.add_argument('--captions_path', type=str, default='captions/', help='s3 path location for generated test captions')
    parser.add_argument('--version', type=str, default='1.0', help='model versioning')
    
    args = parser.parse_args()
    print(args)
    # python3 train.py --num_epochs=2 --batch_size=10 --version=0.1
    main(args)
