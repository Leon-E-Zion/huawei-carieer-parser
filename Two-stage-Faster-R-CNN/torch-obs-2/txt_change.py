



f = open('./2007_train.txt')
lines = f.readlines() #整行读取
f.close()
for rs in lines:
  #  rs = line.rstrip('\n') #去除原来每行后面的换行符，但有可能是\r或\r\n
    orid = r'/home/ray/workspace/Leon/faster-rcnn-torch/faster-rcnn-pytorch-master/'
    new_path = r'/home/work/modelarts/user-job-dir/faster-rcnn-pytorch-master/'
    newname=rs.replace(orid,new_path)
    newfile=open('2007_train_.txt','a')
    newfile.write(newname)
    newfile.close()
