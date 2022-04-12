# Chinese-dialect-recognition-with-efficientNet
pyotrch
大学生的大创项目，识别方言的种类，数据集用的是2018年讯飞方言识别挑战赛的数据集，包括9种方言：长沙话，河北话，合肥话，客家话，南昌话，宁夏话，陕西话，上海话，四川话，
每种方言有6000段pcm格式的录音，每段录音差不多7秒左右。
在processData文件夹中，pcm转为wav格式，再转换为声谱图
在train&predict文件夹中，先将声谱图从灰度转为RGB,再使用efficientNet在pytorch环境中训练，附带一个预测的代码。在我的训练中，使用了efficientNet B0, 9类方言每类250张图片进行图像分类的训练，准确率可达78%。
在GUI文件夹中，制作了一个GUI的界面，可以现场录音，预测方言种类，也可以选择传入wav文件进行预测。
