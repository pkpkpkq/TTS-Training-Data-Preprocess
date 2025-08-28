# TTS-Training-Data-Preprocess
# 输入
Test/

├── 1.wav

├── 1.lab

├── 2.wav

└── 2.lab

名字随意，一组wav和lab需要名字相同
# 输出
output/

├── Test/

│ ├── !{音频总长度}.txt

│ ├── ！合并记录.txt

│ ├── ！切分记录.txt

│ ├── ！Test.log

│ ├── 1.wav （如果USE_LAB_TEXT_AS_FILENAME = True 那么这里是{1.lab的内容}.wav）

│ └── 2.wav （如果USE_LAB_TEXT_AS_FILENAME = True 那么这里是{2.lab的内容}.wav）

└── Test.list
# 使用教程
安装所需库后修改Python.py中的ROOT_DIR为需要处理的音频的目录（不会处理子目录）后运行Python.py即可。
