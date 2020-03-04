==================模型训练说明====================

一、环境安装
1. Linux操作系统
2. 安装CUDA 10.0，用于GPU训练
3. 使用 
	pip install -r requirements.txt
    安装训练所需的库

==============================================

二、使用说明
1. 将数据放到data目录下，名称要求是以_arr.txt和_label.txt结尾的成对文件
2. 运行
	python train.py
    进行模型训练
3. 在models目录下生成完成训练后的模型checkpoint.pth，即
	models/checkpoint.pth