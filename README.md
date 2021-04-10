# Chinese2English-Translator-with-Transformers
-`Hello guys😄, hope you are doing awesome`
-`In this repository, I Improved my previous chinese2english translator with transformer and it do get a tremendous improvement`

# Examples
-'Here are some translating examples, this model has a great accurancy on the training data(overfitting😥), and 
-`it seems to work bad on test data. I think its the problem of datasets because i only used the validation dataset
-`of translation2019zh, which is quite small. But, you know that, i do this just for fun😁. Anyway, if i want and 
-`i have time, i will train a really useful chinese2english translator
- ![test_example](https://github.com/Mountchicken/Image-Captioning-pytorch/blob/main/text_examples/dog.jpg)
- Predicted: <SOS> a brown dog is running on the grass . <EOS>
- <img src="https://github.com/Mountchicken/Image-Captioning-pytorch/blob/main/text_examples/happy.jpg" width="216" height="288" alt="😀"/><br/>
- Predicted: <SOS> a woman in a red shirt and a man in a white shirt smile for the camera . <EOS>
## 项目结构
### 文件

- `Model.py`: 定义Inception_v3模型，LSTM模型
- `get_loader.py`:定义ImageCaptioning数据集
- `Train.py`: 训练模型，建议不要修改超参数，因为我发现好像只有特定的超参数才能有较好的训练效果
- `inferrence`: 测试你自己的图片


### 文件夹
- `archive`: 存放flickr8k数据集，[下载地址](https://www.kaggle.com/aladdinpersson/flickr8kimagescaptions)
- `test_examples`:测试图片
## 如何使用

### 如何训练
#### 1.下载spacy库所需文件
- `pip install spacy`
- `download en_core_web_sm,[download](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.0.0)
- `-pip install 安装包`
#### 2.运行train.py
### 如何测试自己的图片
- `修改inferrence中测试图片地址(23行),运行即可

## 联系方式（获取预训练权重）
- mountchicken@outlook.com


