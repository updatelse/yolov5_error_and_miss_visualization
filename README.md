# 分析yolov4、yolov5、yolov7目标检测模型的误检漏检情况

## 本程序可以实现两项功能
-   对误检、漏检情况进行可视化分析, 并输出到不同的文件夹以便浏览
-   对真正误检的图片生成预标注xml(这里'真正误检'的意思下面会解释)


## 需要准备的数据
-   原始图片
-   原始标注的xml
-   模型推理出来的txt文件
-   类别classes.names(手动编辑一个)

## 数据组织形式
输入:
```
data/
  classes.names     #a labels sequence
  images/
    h136.jpg
    8850.jpg
    AB59.jpg
  xml/
    h136.xml
    8850.xml
    AB59.xml
  labels/
    h136.txt         #model inference, yolo format
    8850.txt
    AB59.txt
```
```
解释classes.names

    person
    helmet
    nohelmet
    others
```
```
解释h136.txt 

    2 0.15799558 0.21358356 0.21856634 0.37601453
    0 0.102568194 0.57960725 0.16702475 0.8333428
    3 0.46533334 0.3801881 0.51561904 0.55205184
```

## 如何得到labels中的每个txt文件：
-  如果是yolov7检测模型, 在python detect.py或者python test.py测试时加上--save-txt参数即可, 会对每张图片推理结果保存成一个 *.txt文件, 这是我们需要的。

-  如果是yolov4(即darknet)检测模型, 在使用video_parser工程推理时, 在main.py程序中的detect_img_file( )函数中添写一个保存结果的小函数, 如下所示：
```
def write_detect_box_to_txt(boxes,filename): #检测出boxes框, 生成对应名称的单个txt
    base_path = './smoke_fire_txt_save_path'
    if not os.path.exists('./smoke_fire_txt_save_path'):
        os.makedirs(base_path)
    list_file = open("{}/{}.txt".format(base_path,filename), 'w')
    for i in range(len(boxes)):   
        for j in range(len(boxes[0])):  
            list_file.write(str(boxes[0][j][6]) + " " + " ".join(str(a) for a in boxes[0][j][:-3]) + '\n')
    list_file.close()
```
```
然后在main.py程序中的detect_img_file()函数里当boxes = g_detector.detect_img(img)预测出boxes后, 调用上面的小函数write_detect_box_to_txt(boxes,filename), 即能生成每个*.txt 。

```

## 误检漏检代码解读
该误检漏检代码的select_pic_by_p_r( )函数中有两种txt读取方式, 按需选择:
```
    #yolov4
    #pre_dict_per = get_txt_result_from_yolov4(ori_xmlfile,classes,img)  #如果是yolov4检测处理来的结果, 用这行读取txt
    #yolov5、yolov7
    pre_dict_per = get_txt_result(ori_txtfile,classes,img)     # 如果是yolov5或yolov7检测处理来的结果, 用这行读取txt
```

## 使用步骤：
在python脚本的main模块中更改你的输入数据路径：

```
names_path = "shared/analyse/data/classes.names",  
picpath = "/shared/analyse/data/images",
ori_xmlpath = "shared/analyse/data/xml",
detect_txtpath = "shared/analyse/data/labels",
savepath = "shared/analyse/data/output_"+classe_names[i],
classes = [classe_names[i]],
iou_thres = 0.25,
need_auto_save_missing_xml = False,                     #这一项是个开关, 目的是保存真正误检的预测框为xml, 实现预标注.
names_path = "/shared/analyse/data/classes.names"
```
```
以及该python脚本的第14行全局变量clsname, 改成你的classes.names中的标签：
clsname = ['person','helmet','nohelmet','others']
```
此脚本是根据labels文件夹每个txt文件去images、xml文件夹寻找同名文件, 所以images、xml文件夹内数量多余labels文件夹内数量没事, 多余的不会用到。

## 输出结果
```
data/
  output_person/
    FP(error_detect)/
    FN(miss_detect)/
  output_helmet/
    FP(error_detect)/
    FN(miss_detect)/
  output_others/
    FP(error_detect)/
    FN(miss_detect)/
  output_nohelmet/
    FP(error_detect)/
        FP_other_GT/
          8850.jpg
          ***.jpg
        FP_background/
          123.jpg
    FN(miss_detect)/
        FN_really/
          AB59.jpg
          ***.jpg
        FN_other_predict_class/
          ***.jpg
```
会生成classes.names中每个类别名称的一级文件夹, 然后下面分误检、漏检二级文件夹

## 自动为误检框生成预标注xml
这里解释一下, 误检分为两种：
- 背景的误检：
    我们本想让模型正确检测出安全帽, 一种是模型把气球检测成了安全帽, 误检; 另一种是模型确实检测出一个安全帽, 但由于我们的xml中对这个安全帽漏标注了, 本脚本识别出这个安全帽是'误检', 
实际上, 针对这个安全帽来说, 并不是误检, 只是因为我们漏标注, 模型能检测出是正确的行为, 因为我们需要对这个安全帽生成标注xml文件, 免除人工标注, 后续放在训练集训练。
- 误检成它类：
    假如现在有人、安全帽、非安全帽待检测, 模型检测出这个目标是安全帽, 但使用该预测框跟xml中所有GT框比较时, 跟nohelmet的GT框交并比最大, 说明模型误检了该目标, 其GT实际是非安全帽, 
(检测出的框是对的, 只是分类分错了)。

如果你想针对背景的误检生成预标注xml文件, 请你在python脚本的main模块将need_auto_save_missing_xml = True ,这样在error_detect文件夹会额外生成预标注xml和原始图片两项文件夹。
```
data/
  output_person/
    FP(error_detect)/
        FP_other_GT/
        FP_background/
        really_error_detect_xml/     # this
        really_error_detect_pic/     # this
    miss_detect/
```

注意：这里生成的预标注xml文件是所有背景的误检, 其中也包括了上面提到的'气球检测成安全帽',因此这些xml不能直接拿来使用, 得用labelimg复核一下剔除掉'气球检测成安全帽'这些, 留下真正的安全帽漏标。
