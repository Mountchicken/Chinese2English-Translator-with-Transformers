import torch
import spacy
from utils import load_stoi, load_itos
from model import Transformer
import time

def translate(model, sentence, chinese_vocab, english_vocab, device, max_length=50):
    # Load chinese tokenizer
    spacy_ch= spacy.load("zh_core_web_sm")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ch(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, '<sos>')
    tokens.append('<eos>')

    # Go through each german token and convert to an index
    text_to_indices = [chinese_vocab['stoi'][token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english_vocab['stoi']["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english_vocab['stoi']["<eos>"]:
            break

    translated_sentence = [english_vocab['itos'][idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]

if __name__=="__main__":
    #load vocab
    print('===> loading vocabulary')
    chinese_itos = load_itos('saved_vocab/chinese_itos.txt')
    english_itos = load_itos('saved_vocab/english_itos.txt')
    chinese_stoi = load_stoi('saved_vocab/chinese_stoi.txt')
    english_stoi = load_stoi('saved_vocab/english_stoi.txt')
    chinese_vocab={'stoi':chinese_stoi,'itos':chinese_itos}
    english_vocab={'stoi':english_stoi,'itos':english_itos}

    # Model hyperparamters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_vocab_size = len(chinese_stoi)
    trg_vocab_size = len(english_stoi)
    embedding_size = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dropout = 0.1
    max_len = 120 #最长一个句子的长度也不能超过 max_len
    forward_expansion = 2048 #pytorch官方实现的transformer中，这个参数就是线性层升维后的结果
    src_pad_idx = chinese_stoi['<pad>']
    trg_pad_idx = english_stoi['<pad>']

    #initialize model
    print('==>Initializng model')
    model = Transformer(
        embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads,
        num_encoder_layers, num_decoder_layers, forward_expansion, dropout, max_len, device
    ).to(device)

    #Load model
    print('==>Loding model')
    param = torch.load('my_checkpoint.pth.tar')["state_dict"]
    model.load_state_dict(param)
    model.eval()

    #test sentence
    print('==>Translating')
    sentence = "我的女朋友知道如何去倾听和帮助。"
    time1 = time.time()
    trans = translate(model, sentence, chinese_vocab, english_vocab, device, max_length=50)
    print('Chinese: '+sentence)
    print('English: '+' '.join(trans))
    print('Translating time: {}'.format(time.time() - time1))