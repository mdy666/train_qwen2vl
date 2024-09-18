# package
请更新相关包

pip install --upgrade accelerate

pip install --upgrade deepspeed

# 训练
主要针对图片进行训练，我以flicker8k为例，建议4张A100以上，使用deepspeed进行训练。

如果想要更好的训练效果，建议改下参数之类的，我的程序只是能跑起来

# 启动命令
bash run.sh

# 数据集格式
数据集格式建议是下图这样
![image](https://github.com/user-attachments/assets/9c27bde7-7c6d-4e0d-bff4-ec928c2232eb)

请自行在dataset中进行处理，即self.data[i]， 无论多少轮，多少张图片都可以进行处理，建议使用json line文件

CustomProcessor接收messages进行处理
