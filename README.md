# Chinese2English-Translator-with-Transformers
- Hello guys😄, hope you are doing awesome
- In this repository, I Improved my previous chinese2english translator with transformer and it do get a tremendous improvement

# Examples
- Here are some translating examples, this model has a great accurancy on the training data(overfitting😥), and 
- it seems to work bad on test data. I think its the problem of datasets because i only used the validation dataset
- of translation2019zh, which is quite small. But, you know that, i do this just for fun😁. Anyway, if i want and 
- i have time, i will train a really useful chinese2english translator
- ![test_example](https://github.com/Mountchicken/Chinese2English-Translator-with-Transformers/tree/main/Image_github/one.png)
- ![test_example](https://github.com/Mountchicken/Chinese2English-Translator-with-Transformers/tree/main/Image_github/two.png)
- ![test_example](https://github.com/Mountchicken/Chinese2English-Translator-with-Transformers/tree/main/Image_github/three.png)
- ![test_example](https://github.com/Mountchicken/Chinese2English-Translator-with-Transformers/tree/main/Image_github/four.png)
- Predicted: <SOS> a brown dog is running on the grass . <EOS>
- <img src="https://github.com/Mountchicken/Image-Captioning-pytorch/blob/main/text_examples/happy.jpg" width="216" height="288" alt="😀"/><br/>
- Predicted: <SOS> a woman in a red shirt and a man in a white shirt smile for the camera . <EOS>
- 
# Requirements
- `torchtext >= 1.8'
- `spacy`
# Structure
## Files
- `get_loaderl.py`: Define the dataloader using torchtext
- `train.py`: Train the model
- `model.py`: Define the model
- `inference.py`:Translate your own chinese sentece to english one !!
## Folders
- 'saved_vocab`: Contain serval vocabulary txt and you can also generate then during training
- `translation2019zh`: This is Google's chinese2english translation samples. It's huge and i only take the validation dataset to train

# How to use
## How to train
- 'Go inside the train.py, set some hyperparameters if you want or just run it!'
- 
## How to translate my own sentence
- `Go inside the inference.py, set the your own chinese sentence at line 73 

# Contact me for trained_weights(too big to upload)
- mountchicken@outlook.com
