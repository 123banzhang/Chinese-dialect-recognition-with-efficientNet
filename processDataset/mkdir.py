def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


# 定义要创建的目录
# mkpath = "C:/Users/Pang Bo/Desktop/大三下/方言/Spoken-language-identification-pytorch/images/speaker0"
# 调用函数
# for i in range(31, 36):
#     mkpath = "C:/Users/Pang Bo/Desktop/大三下/方言/Spoken-language-identification-pytorch/images/speaker"+str(i)
#     mkdir(mkpath)
for i in range(1, 10):
    mkpath = "C:/Users/Pang Bo/Desktop/大三下/方言/Spoken-language-identification-pytorch/images/speaker0"+str(i)
    mkdir(mkpath)
for i in range(10, 31):
    mkpath = "C:/Users/Pang Bo/Desktop/大三下/方言/Spoken-language-identification-pytorch/images/speaker"+str(i)
    mkdir(mkpath)
