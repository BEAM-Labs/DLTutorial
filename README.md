# DLTutorial
新学生深度学习入门，可以按照顺序和需求自行安排、调节进度。

# 1. 入门功课、tutorial等
## A 基础篇
## A.1. Andrew Ng 的 Deep Learning Specialization及课后作业
视频地址： https://www.bilibili.com/video/BV1pJ41127Q2  
HW: https://github.com/ppx-hub/deep-learning-specialization-all-homework/tree/main/Homework-NoAnswer  
HW solution: https://github.com/amanchadha/coursera-deep-learning-specialization 

- S1: Neural Networks and Deep Learning 
  - video: from **P9 - 1: What is a neural network** to **P50 - 8: What does this have to do with the brain?**
  - topics covered: logistic regression, computational graph, activation function, backpropagation and etc

- S2: Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization
  - video: from **P51 - 1: Train / Dev / Test sets** to **P85 - 11: TensorFlow**, 注：现在均使用pytorch
  - topics covered: bias variance tradeoff, regularization, dropout, gradient descent(Momentum, RMSprop, Adam), learning rate decay, batch normalization and etc

- S3: Structuring Machine Learning Projects
  - video: from **P86 -1 : Why ML Strategy** to **P107 - 10: Whether to use End-to-end Deep Learning**
  - topics covered: transfer learning, multi-task learning and etc
  - 这个好像是没有作业的？
 
- S4: Convolutional Neural Networks
  - video: from **P108 - 1: Computer Vision** to **P150 - 11: 1D and 3D Generalizations**
  - topics covered: CNN basics (padding, pooling and etc), ResNet, Data augmentation, YOLO, U-Net, Siamese Network and etc

- S5: Sequence Models
  - video: from **P151 - 1: Why Sequence Models?** to **P180 - 8: Attention Model**
  - video: transformer network部分我在b站没有找到视频，有找到的同学可以补充，youtube视频可以参考这里：https://www.youtube.com/watch?v=S7oA5C43Rbc&t=18037s， 时间大概从5小时左右开始
  - topics covered: GRU, LSTM, word2vec, glove, beam search, attention model, transformer and etc
 
## A.2. 李宏毅机器学习
- B站2023年课程链接：https://www.bilibili.com/video/BV1TD4y137mP
- B站2024年课程链接（主要为生成式AI，如GPT和Diffusion）：https://www.bilibili.com/video/BV1BJ4m1e7g8
- 2023年课程网页：https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php
- 2024年课程网页：https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php
- 最大优点是课程网页上有代码作业，课程上难度偏简单，主要以讲解概念为主。

## A.3. 动手学强化学习
- 课程连接：https://hrl.boyuai.com/chapter/intro/
- 视频连接：https://www.boyuai.com/elites/course/xVqhU42F5IDky94x
- 风格与《动手学深度学习》类似，但不是李沐团队的课程。上交老师出品，内容还是不错的。



## B. 深入篇 :
## B.1. UCB cs182: Designing, Visualizing and Understanding Deep Neural Networks

- UCBerkely: https://cs182sp21.github.io
- 课程作业官网+github
- 视频b站历年

 ## B.2. UCB cs285: Deep Reinforcement Learning

- UCBerkely: [https://cs182sp21.github.io](https://rail.eecs.berkeley.edu/deeprlcourse/)
- 课程作业官网+github
- 视频b站历年




# 2. 经典入门文章，所有研究方向必读
## 2.1. transformer 架构细节学习
- transformer比较重要，请首先阅读[原文](https://arxiv.org/abs/1706.03762)，没有看懂的部分记录下来
- 完成 https://www.bilibili.com/video/BV1pu411o7BE/， 看是否能够回答之前记录的问题， 并找出自己阅读论文时候漏掉的关键点
- 阅读 [illustrated transformer](https://jalammar.github.io/illustrated-transformer/)
- 完成 [annotated transformer](http://nlp.seas.harvard.edu/annotated-transformer/)，熟悉代码细节

  
## 2.2. Pre-trained Language Models (PLM)
limu paper reading repo: https://github.com/mli/paper-reading
- 2.2.1. 阅读下列预训练语言模型文章: GPT, BERT, GPT2, 请**先自己阅读**，之后和[这里](https://github.com/mli/paper-reading)进行对比，比较下自己是否漏读了重要内容
  - **Improving Language Understanding by Generative Pre-Training**. *Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever*. Preprint. [[pdf](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)] [[project](https://openai.com/blog/language-unsupervised/)] (**GPT**)
  - **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. *Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova*. NAACL 2019. [[pdf](https://arxiv.org/pdf/1810.04805.pdf)] [[code & model](https://github.com/google-research/bert)]
  - **Language Models are Unsupervised Multitask Learners**. *Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever*. Preprint. [[pdf](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)] [[code](https://github.com/openai/gpt-2)] (**GPT-2**)
  - **RoBERTa: A Robustly Optimized BERT Pretraining Approach**. *Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov*. Preprint. [[pdf](https://arxiv.org/pdf/1907.11692.pdf)] [[code & model](https://github.com/pytorch/fairseq)] (optional)

- 2.2.2. 阅读seq2seq语言模型相关文章
  - **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer**.  *Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu*. Preprint. [[pdf](https://arxiv.org/pdf/1910.10683.pdf)] [[code & model](https://github.com/google-research/text-to-text-transfer-transformer)] (**T5**)
  - **mT5: A massively multilingual pre-trained text-to-text transformer**. *Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, Colin Raffel* Preprint. [[pdf](https://arxiv.org/abs/2010.11934)](**mT5**)

- 2.2.3. 建议完成语言模型的pre-train (demo)和fine-tuning 
  - 请**不要**直接调用huggingface中[run_glue.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py)里的trainer进行fine-tune， 但可以使用其下载和load data
  - 也可以利用[script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)自行下载[数据](https://gluebenchmark.com/tasks)， 
  - fine-tune language models (BERT, GPT, RoBERTA, T5 and etc) on GLUE benchmark (MRPC和RTE数据集较小，可以优先只考虑这两个数据集)，注意调整超参数用来获得更好的结果
  - 在[wikitext](https://huggingface.co/datasets/wikitext)上pre-train 自己的语言模型，在MRPC和RTE数据集上fine-tune, 并和BERT的结果进行比较


## 2.3. Multi-modal, diffusion, LLM and etc (TODO)


# 3. 根据研究方向的细分文章 (TODO)
## 3.1. Bio Language Models
## 3.2. Structure Prediction
## 3.3. Proteomics 
## 3.4. Design and et al.  


# 4. Miscellaneous
## 4.1. Good informative talks
- [Stanford CS25: V4 I Jason Wei & Hyung Won Chung of OpenAI](https://www.youtube.com/watch?v=3gb-ZkVRemQ)：讲叙一些基本原理和思考，强烈建议
- [Yannic Kilcher](https://www.youtube.com/@YannicKilcher), 他的文章讲解方式是非常好的，值得学习。例如 [transformer](https://www.youtube.com/watch?v=iDulhoQ2pro), [Sparse Expert Models](https://www.youtube.com/watch?v=ccBMRryxGog&list=LL&index=21), [switch transformer](https://www.youtube.com/watch?v=ccBMRryxGog&list=LL&index=21) and etc. 
