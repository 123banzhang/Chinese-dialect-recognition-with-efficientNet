import shutil,os,re
new_path='C:\\Users\\Pang Bo\\Desktop\\train\\sichuan'
for derName, subfolders, filenames in os.walk('C:\\Users\\Pang Bo\\Desktop\\大三下\\方言\\Spoken-language-identification-pytorch\\images'):
#print(derName/subfolders/filenames)
    for i in range(len(filenames)):
        if filenames[i].endswith('.png'):
            file_path=derName+'\\'+filenames[i]
            newpath=new_path+'\\'+filenames[i]
            shutil.copy(file_path,newpath)