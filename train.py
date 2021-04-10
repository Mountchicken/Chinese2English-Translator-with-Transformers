import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from get_loader import get_loader
from model import Transformer

def train():
    # Traing
    num_epochs = 300
    learning_rate = 0.0001
    batch_size = 64

    #get trainloader and vocab
    save_vocab = True #是否保存词典，建议第一次训练的时候保存，后面就不需要保存了
    train_iterator, english, chinese = get_loader(batch_size, save_vocab)

    # Model hyperparamters
    load_model = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_vocab_size = len(chinese.vocab)
    trg_vocab_size = len(english.vocab)
    embedding_size = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout = 0.1
    max_len = 120 #最长一个句子的长度也不能超过 max_len
    forward_expansion = 2048 #pytorch官方实现的transformer中，这个参数就是线性层升维后的结果
    src_pad_idx = chinese.vocab.stoi['<pad>']
    trg_pad_idx = english.vocab.stoi['<pad>']
    #Tensor board
    writer = SummaryWriter(f'runs/loss_plot')
    step = 0


    # Initialize network
    model = Transformer(
        embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads,
        num_encoder_layers, num_decoder_layers, forward_expansion, dropout, max_len, device
    ).to(device)

    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, factor=0.1, patience=15, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx)

    if load_model:
        load_checkpoint(torch.load('my_checkpoint.pth.tar'),model, optimizer)
    sentence = "你知道的，我会永远爱着你。" #测试用句


    for epoch in range(num_epochs):
        print(f'Epoch [{epoch}]/[{num_epochs}]')
        checkpoint = {'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict()}
        save_checkpoint(checkpoint)

        model.eval()
        translate = translate_sentence(model, sentence, chinese, english, device, max_length=50)
        print(f'Translated example sentence \n {translate}')
        model.train()

        loop = tqdm(enumerate(train_iterator),total=len(train_iterator),leave=False)
        epoch_loss = 0
        count = 0
        for batch_idx, batch in loop:
            count = count + 1
            inp_data = batch.ch.to(device)
            target = batch.eng.to(device)
            output = model(inp_data, target[:-1,:])
            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin.
            # Let's also remove the start token while we're at it
            output = output.reshape(-1,output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)
            epoch_loss += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1) #梯度裁剪
            optimizer.step()
            writer.add_scalar('batch Loss', loss, global_step=step)
            step += 1
            loop.set_description(f'Epoch[{epoch}/{num_epochs}]')

        mean_loss = epoch_loss / count
        scheduler.step(mean_loss)
        writer.add_scalar('mean Loss', mean_loss, epoch)
        print("mean loss: {}".format(mean_loss))

if __name__=="__main__":
    train()