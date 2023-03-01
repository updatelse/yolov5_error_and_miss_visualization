from lib2to3.pgen2.pgen import generate_grammar
from operator import ge
import xml.etree.ElementTree as ET
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import os
import codecs
import io
import numpy as np
import cv2
import copy

#clsname = ['person','helmet','nohelmet','others']
clsname = ['person','safety_belt','blue_gray','red','fgy_g','fgy_r','other_clothes']

def append_object(name, xmin, ymin, xmax, ymax, top):   
    object_item = SubElement(top, 'object')
    name_item = SubElement(object_item, 'name')
    name_item.text = name

    pose = SubElement(object_item, 'pose')
    pose.text = "Unspecified"

    truncated = SubElement(object_item, 'truncated')
    truncated.text = "0"

    difficult = SubElement(object_item, 'difficult')
    difficult.text = str(0)

    bnd_box = SubElement(object_item, 'bndbox')
    x_min = SubElement(bnd_box, 'xmin')
    x_min.text = str(int(xmin))
    y_min = SubElement(bnd_box, 'ymin')
    y_min.text = str(int(ymin))
    x_max = SubElement(bnd_box, 'xmax')
    x_max.text = str(int(xmax))
    y_max = SubElement(bnd_box, 'ymax')
    y_max.text = str(int(ymax))

def generate_label(img, boxes, class_names=None, filename = None, save_path_xml = None,save_path_pic = None):    
    
    top = Element('annotation')
    folder = SubElement(top, 'folder')
    folder.text = 'pic'

    only_name = os.path.basename(filename)
    image_cv2_info = cv_imread(filename) 
    et_filename = SubElement(top, 'filename')
    et_filename.text = only_name
    
    local_img_path = SubElement(top, 'path')
    local_img_path.text = filename

    source = SubElement(top, 'source')
    database = SubElement(source, 'database')
    database.text = 'Unknown'
        
    size_part = SubElement(top, 'size')
    et_width = SubElement(size_part, 'width')
    et_height = SubElement(size_part, 'height')
    et_depth = SubElement(size_part, 'depth')
    et_width.text = str(img.shape[1])
    et_height.text = str(img.shape[0])
    et_depth.text = str(3)
    
    segmented = SubElement(top, 'segmented')
    segmented.text = '0'

    width = img.shape[1]
    height = img.shape[0]

    if len(boxes) > 0:
        print('len(boxes):', len(boxes))
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int(box[0])
        if x1 < 0:
            x1 = 0
        y1 = int(box[1])
        if y1 < 0:
            y1 = 0
        x2 = int(box[2])
        if x2 > width:
            x2 = width - 1
        y2 = int(box[3])
        if y2 >= height:
            y2 = height - 1

        append_object(class_names, x1, y1, x2, y2, top)

    rough_string = ElementTree.tostring(top, 'utf8')
    root = etree.fromstring(rough_string)
    result_str= etree.tostring(root, pretty_print=True, encoding='utf-8').replace("  ".encode(), "\t".encode())
    #这里要设置xml输出路径了
    xml_path = "{}/{}.xml".format(save_path_xml,os.path.splitext(os.path.basename(filename))[0])
    cv_imwrite(os.path.join(save_path_pic, os.path.basename(filename)), image_cv2_info)
    out_file = codecs.open(xml_path,  'w', encoding= 'utf-8')
    out_file.write(result_str.decode('utf8'))
    out_file.close()

def cv_imread(file_path=""):
    img_mat = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 1)
    #img_mat = cv2.imread(file_path)
    return img_mat

def cv_imwrite(saved_img_path, img, ext = '.jpg'):
    cv2.imencode(ext, img)[1].tofile(saved_img_path) 

def get_file_path(base_path,name):
    img_suffixes=['jpg','png','JPG','PNG','jpeg','JPEG']
    for i in img_suffixes:
        file_path = "{}/{}.{}".format(base_path, name, i)
        if os.path.exists(file_path):
            return file_path

def get_classes(names_path):  #从一个labels.txt文件中读取类别标签名字
    classes=[]
    try:
        fi = open(names_path, "r", encoding='UTF-8')
    except:
        fi = io.open(names_path, "r", encoding='UTF-8')
    for line in fi.readlines():
        line = line.strip('\n')
        classes.append(line)
    fi.close()
    return classes

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def compute_iou(box1, box2, wh=False):
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0]-box1[2]/2.0), int(box1[1]-box1[3]/2.0)
        xmax1, ymax1 = int(box1[0]+box1[2]/2.0), int(box1[1]+box1[3]/2.0)
        xmin2, ymin2 = int(box2[0]-box2[2]/2.0), int(box2[1]-box2[3]/2.0)
        xmax2, ymax2 = int(box2[0]+box2[2]/2.0), int(box2[1]+box2[3]/2.0)

    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    area1 = (xmax1-xmin1) * (ymax1-ymin1)
    area2 = (xmax2-xmin2) * (ymax2-ymin2)
    inter_area = (np.max([0, xx2-xx1])) * (np.max([0, yy2-yy1]))
    iou = inter_area / (area1+area2-inter_area+1e-6)

    return iou

def read_xml_file(xml_file,classes):
    label_dict={}
    obj_dict={}
    xml_ =codecs.open(xml_file,"r","utf-8")
    tree = ET.parse(xml_)
    root = tree.getroot()
    size = root.find('size')
    #filename=root.find('filename').text
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    img_id=os.path.splitext(os.path.basename(xml_file))[0]#os.path.basename(filename)
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes:
            continue
        #cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')       
        b = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)]
        if not obj_dict.get(cls):
            obj_dict[cls]=[]
        obj_dict[cls].append(b)

    label_dict[img_id]=obj_dict
    
    return label_dict

def get_txt_result(xml_file,classes,img):    #这里xml_file是使用yolo工程detect.py输出的预测值txt,  eg: 20229117.txt, apm3bb.txt, 反归一化
    obj_dict={}
    if os.path.exists(xml_file):
        resultdata = open(xml_file,"r",encoding="utf-8").readlines()
        w = img.shape[1]
        h = img.shape[0]
        for i in resultdata:
            i = i.strip()
            results= i.split(' ')
            x = float(results[1]) * w
            y = float(results[2]) * h
            bw = float(results[3]) * w
            bh = float(results[4]) * h
            #conf = float(results[5])
            cls = int(results[0])
            cls = clsname[cls]
            xmin = x - bw//2
            xmax = x + bw //2
            ymin = y - bh //2
            ymax  = y + bh //2
            b = [xmin,ymin,xmax,ymax]
            if not obj_dict.get(cls):
                obj_dict[cls]=[]
            obj_dict[cls].append(b)
            #obj_dict[cls].append(b)
    else:
        for cls in classes:
            obj_dict[cls]=[]       
    return obj_dict

def get_txt_result_from_yolov4(xml_file,classes,img):
    obj_dict={}
    if os.path.exists(xml_file):
        resultdata = open(xml_file,"r",encoding="utf-8").readlines()
        w = img.shape[1]
        h = img.shape[0]
        for i in resultdata:
            i = i.strip()
            results= i.split(' ')
            xmin = float(results[1]) * w
            ymin = float(results[2]) * h
            xmax = float(results[3]) * w
            ymax = float(results[4]) * h
            #conf = float(results[5])
            cls = int(results[0])
            cls = clsname[cls]
            b = [xmin,ymin,xmax,ymax]
            if not obj_dict.get(cls):
                obj_dict[cls]=[]
            obj_dict[cls].append(b)
    else:
        for cls in classes:
            obj_dict[cls]=[]       
    return obj_dict


def get_obj_dict_per(xml_file,classes):
    obj_dict={}
    if os.path.exists(xml_file):
        xml_ =codecs.open(xml_file,"r","utf-8")
        tree = ET.parse(xml_)
        root = tree.getroot()
        size = root.find('size')
        #filename=root.find('filename').text
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        img_id=os.path.splitext(os.path.basename(xml_file))[0]
        for obj in root.iter('object'):
            # difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes:
                continue
            #cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')       
            b = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)]
            if not obj_dict.get(cls):
                obj_dict[cls]=[]
            obj_dict[cls].append(b)
    else:
        for cls in classes:
            obj_dict[cls]=[]       
    return obj_dict

def get_xml_file(imgfile,xmlpath):
    #img_id=os.path.basename(imgfile)[:-4]
    tmpName=os.path.basename(imgfile).split('.')
    img_id=''
    for i in range(len(tmpName)-1):
        img_id=img_id+tmpName[i]+'.'
    xmlfile=os.path.join(xmlpath,img_id+'xml')
    return xmlfile if os.path.exists(xmlfile) else None

def get_label_dict(xmlpath,classes):
    label_dict={}
    filelist=os.listdir(xmlpath)
    for file in filelist:
        xmlfile =os.path.join(xmlpath,file)
        dict_per=read_xml_file(xmlfile,classes)
        label_dict.update(dict_per)
    return label_dict


def save_err_img(obj_predict, ori_imgfile, xmlpath, classes, errDir, iou_thres=0.5,need_auto_save_missing_xml =True, names_path=None):
    
    xmlfile = get_xml_file(ori_imgfile, xmlpath)              # 获取xmlfile，返回的是绝对路径，不是xml里面的值
    obj_label = get_obj_dict_per(xmlfile, classes)            # 从xmlfile中获取类别的G-T数据
    img = cv_imread(ori_imgfile)                              # 读取图像，ori_imgfile是原始图片,还未使用cv2读取
    ################################
    img_err_but_have_other_GT = copy.deepcopy(img)            # 这张图专门用来画误检框，有GT
    img_err_background = copy.deepcopy(img)                   # 这张图专门用来画真正误检框，无GT
    img_miss_background = copy.deepcopy(img)                  # 这张图专门用来画漏检框
    img_miss_with_other_predict = copy.deepcopy(img)          # 这张图专门用来画漏检，但是被其他类检出了的图
    for_generate_xml = []
    ###############################
    err_flag =False                                           # 误检标志，用来标记误检但有其它G-T真值的检测框
    err_flag_really = False                                   # 误检标注，用来标记误检，是真正背景类误检，没有G-T真值
    miss_flag = False                                         # 漏检标志，用来标记漏检，是真正漏检
    miss_flag_other_predict = False                           # 漏检标注，用来标注漏检但是在其它类别中有这个GT框的预测框

    # 遍历所有类别，按照类别来检索漏检、误检
    for cls in classes:                                       # 传递过来的，第一个处理的是person
        
        boxes = obj_label.get(cls)                            # 获取对应类别的gt
        pre_boxes = obj_predict.get(cls)                      # 获取对应类别的预测框
        # 如果有预测框，但是没有gt，说明是这张图出现了误检，误检置为true
        # 如果有预测框并且有gt，遍历预测框和gt计算交并比，如果交并比小于设定的阈值则认为是误检
        # 遍历gt，计算其和所有预测框的交并比，如果小于阈值则说明没有该gt没有预测框与之相匹配，漏检，画红框
        if pre_boxes:
            if boxes:
                for idx_pre in range(0, len(pre_boxes)):       # 遍历所有预测框, 计算预测框和所有gt框的交并比，并选择
                    max_iou = 0 
                    for idx in range(0, len(boxes)):           # 计算当前类别第一个预测框 和 当前类别所有gt框的iou
                        tmp_iou = compute_iou(boxes[idx], pre_boxes[idx_pre])
                        if tmp_iou > max_iou:
                            max_iou = tmp_iou               
                    if max_iou < float(iou_thres):             # 如果交并比最大的还小于阈值，当前这个预测框没有GT与之对应上，该预测框是误检                   
                        for k in range(0,len(clsname)):        # 知道该预测框是误检的预测框, 再判断这一块预测矩形区域在xml中有没有GT类别
                            maxiou = 0
                            obj_label_every = get_obj_dict_per(xmlfile, clsname[k])
                            if clsname[k] not in obj_label_every.keys():
                                continue
                            all_boxes = obj_label_every.get(clsname[k])
                            for index in range(0, len(all_boxes)):                             # 寻找当前误检的这个预测框跟xml中哪个GT框最接近，找其实际类别名
                                tmp_iou = compute_iou(all_boxes[index], pre_boxes[idx_pre])    # GT框在循环变, 预测框不变，找出最大iou的
                                if tmp_iou > maxiou:
                                    maxiou = tmp_iou
                            if maxiou >= float(iou_thres):                                     # 找到了！ 找到这一块预测矩形区域的实际类别名
                                err_flag=True
                                print("当前预测到了{}类别, xml中也有{}类别, 但一个都没配对上, 判定误检, 经计算, 这块预测矩形区域实际GT类别是{}".format(cls,cls, clsname[k]))
                                #红色的GT is ***
                                cv2.putText(img_err_but_have_other_GT, "GT is " + clsname[k], (int(pre_boxes[idx_pre][2] + 10), int(pre_boxes[idx_pre][1] + (pre_boxes[idx_pre][3]-pre_boxes[idx_pre][1])//2 + 10)), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA) # 误检类别名，画上绿色
                                #绿色的预测矩形框
                                cv2.rectangle(img_err_but_have_other_GT, (int(pre_boxes[idx_pre][0]), int(pre_boxes[idx_pre][1])), (int(pre_boxes[idx_pre][2]), int(pre_boxes[idx_pre][3])), (0, 255, 0), 1)
                                #绿色的预测类别名称
                                cv2.putText(img_err_but_have_other_GT, cls, (int(pre_boxes[idx_pre][0]), int(pre_boxes[idx_pre][1]) - 2), 0, 1, [0, 255, 0], thickness=2, lineType=cv2.LINE_AA)

                            else:
                                err_flag_really = True     # 寻找当前误检的这个预测框跟xml中哪个GT框最接近，找其实际类别名，但是没找到, 说明这个预测框是背景误检
                                print("当前预测到了{}类别, xml中也有{}类别, 但一个都没配对上, 判定误检, 计算后是真正的背景误检".format(cls, cls))
                                if len(for_generate_xml) > 0:                                   
                                    if for_generate_xml[len(for_generate_xml)-1] == pre_boxes[idx_pre]:
                                        #print("出现重复,不加当前框")
                                        continue
                                    else:
                                        #print("不重复, 将此框加到列表中")
                                        for_generate_xml.append(pre_boxes[idx_pre])
                                else:
                                    #print("这里是第一次把预测框转换成xml加到列表中")
                                    for_generate_xml.append(pre_boxes[idx_pre])
                                
                                #print("for_generate_xml_1= ", for_generate_xml)  
                                #蓝色的预测矩形框                          
                                cv2.rectangle(img_err_background, (int(pre_boxes[idx_pre][0]), int(pre_boxes[idx_pre][1])), (int(pre_boxes[idx_pre][2]), int(pre_boxes[idx_pre][3])), (255, 0, 0), 1)
                                #蓝色的预测类别名称
                                cv2.putText(img_err_background, cls, (int(pre_boxes[idx_pre][0]), int(pre_boxes[idx_pre][1]) - 2), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)#明天要改这里
                      
                    else:   # 即, 存在一个iou大于阈值，则当前预测框能在gt框中找到对应，预测正确！ 执行下一个预测框的对比。
                        pass
                for idx in range(0, len(boxes)):               # 遍历所有gt框
                    max_iou = 0
                    for idx_pre in range(0, len(pre_boxes)):
                        tmp_iou = compute_iou(boxes[idx], pre_boxes[idx_pre])    # GT框不变, 预测框循环变
                        if tmp_iou > max_iou:
                            max_iou = tmp_iou
                    if max_iou >= float(iou_thres):                              # 如果有某个预测框大于阈值, 这个gt框匹配上某个预测框，这个gt框没被漏检
                        pass
                    else:                                                        # 该gt框没有匹配上任何一个预测框，这个gt框漏检了，并且是真实漏检，没有其它类别预测
                        miss_flag = True                                         # 真正漏检给它绘上蓝色
                        #绿色的矩形框
                        cv2.rectangle(img_miss_background, (boxes[idx][0], boxes[idx][1]), (boxes[idx][2], boxes[idx][3]), (255, 0, 0), 1)
                        #绿色的gt类别名称
                        cv2.putText(img_miss_background, cls, (boxes[idx][0], boxes[idx][1] - 2), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA) 
                        
            else:                                                       # 表示有预测框，但没有gt，代表这个预测框是 误检
                for idx_pre in range(0, len(pre_boxes)):                # 开始计算这个误检的预测框区域有没有实际GT类别
                    for k in range(0,len(clsname)):
                            maxiouiou = 0
                            obj_label_every = get_obj_dict_per(xmlfile, clsname[k])
                            if clsname[k] not in obj_label_every.keys():
                                continue
                            all_boxes = obj_label_every.get(clsname[k])
                            #print("------",all_boxes)
                            for index in range(0, len(all_boxes)):
                                tmp_iou = compute_iou(all_boxes[index], pre_boxes[idx_pre])
                                if tmp_iou > maxiouiou:
                                    maxiouiou = tmp_iou
                            if maxiouiou >= float(iou_thres):           # 这个误检预测框区域跟xml中的另一个类别的区域重合的很大，于是，该区域有实际GT
                                err_flag = True
                                print("当前预测到了{}类别, xml中没有读取到{}类别, 误检, 这块预测矩形区域实际GT类别是{}".format(cls,cls,clsname[k]))
                                #红色的GT is ***
                                cv2.putText(img_err_but_have_other_GT, "GT is " + clsname[k], (int(pre_boxes[idx_pre][2] + 10), int(pre_boxes[idx_pre][1] + (pre_boxes[idx_pre][3]-pre_boxes[idx_pre][1])//2 + 10)), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA) #
                                #绿色矩形框
                                cv2.rectangle(img_err_but_have_other_GT, (int(pre_boxes[idx_pre][0]), int(pre_boxes[idx_pre][1])), (int(pre_boxes[idx_pre][2]), int(pre_boxes[idx_pre][3])), (0, 255, 0), 1)
                                #绿色误检预测类别名称
                                cv2.putText(img_err_but_have_other_GT, cls, (int(pre_boxes[idx_pre][0]), int(pre_boxes[idx_pre][1]) - 2), 0, 1, [0, 255, 0], thickness=2, lineType=cv2.LINE_AA) 
                            else:
                                err_flag_really = True                   # 这个误检预测框区域跟xml中没有很大的重合区域，属于真正的背景误检
                                print("当前预测到了{}类别, xml中没有读取到{}类别, 误检, 经计算, 是真正的背景误检".format(cls, cls))
                                if len(for_generate_xml) > 0:                                   
                                    if for_generate_xml[len(for_generate_xml)-1] == pre_boxes[idx_pre]:
                                        #print("出现重复,不加当前框_2")
                                        continue
                                    else:
                                        #print("不重复, 将此框加到列表中_2")
                                        for_generate_xml.append(pre_boxes[idx_pre])
                                else:
                                    #print("这里是第一次把预测框转换成xml加到列表中_2")
                                    for_generate_xml.append(pre_boxes[idx_pre])
                                #print("for_generate_xml_2= ", for_generate_xml)
                                # 蓝色的矩形框
                                cv2.rectangle(img_err_background, (int(pre_boxes[idx_pre][0]), int(pre_boxes[idx_pre][1])), (int(pre_boxes[idx_pre][2]), int(pre_boxes[idx_pre][3])), (255, 0, 0), 1)
                                # 蓝色的误检预测类别名称
                                cv2.putText(img_err_background, cls, (int(pre_boxes[idx_pre][0]), int(pre_boxes[idx_pre][1]) - 2), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)
                                             
        else:                # 上面if pre_boxes为假, 该样本的该类别没有预测框, 执行到这。
            if boxes:        # 再看有没有gt框，如果有则说明该类别漏检了，如果没有, 就没事。
                for i in range(0,len(clsname)):
                    if clsname[i] not in obj_predict.keys():
                        continue
                    pre_boxes = obj_predict.get(clsname[i])                             # 挨个读取其余每个类别的预测框            
                    if clsname[i] != cls:
                        for idx_pre in range(0, len(pre_boxes)):
                            max_iou = 0
                            for idx in range(0, len(boxes)):
                                tmp_iou = compute_iou(boxes[idx], pre_boxes[idx_pre])   # 当前类别的gt框跟其余类别的预测框做对比
                                if tmp_iou > max_iou:
                                    max_iou = tmp_iou                         
                            if max_iou >= float(iou_thres):                             # 如果交并比最大的大于阈值,该gt框不但漏检，而且被其它预测类别误检。
                                miss_flag_other_predict=True                            # 置为True，该gt框漏检,最大iou的那个类别误检, 单独输出一下
                                print("没有预测到{}类别, xml中读到了{}类别, GT漏检, 并且GT矩形区域被其它{}类别误检了".format(cls,cls,clsname[i]))
                                #绿色的误检预测类别名称
                                cv2.putText(img_miss_with_other_predict, clsname[i], (boxes[idx][0], boxes[idx][1] - 30), 0, 1, [0, 255, 0], thickness=2, lineType=cv2.LINE_AA) # 误检类别名，画上绿色
                                #红色的gt矩形框
                                cv2.rectangle(img_miss_with_other_predict, (boxes[idx][0], boxes[idx][1]), (boxes[idx][2], boxes[idx][3]), (0, 0, 255), 1) 
                                #红色的gt类别名称
                                cv2.putText(img_miss_with_other_predict, cls, (boxes[idx][0], boxes[idx][1] - 2), 0, 1, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)   #实际GT值，画上红色
                            else:
                                miss_flag = True                                         # 当前类别的gt框跟其余类别的预测框做对比，没有匹配上,当前GT类别仅是漏检。
                                for idx in range(0, len(boxes)):
                                    #蓝色的gt矩形框
                                    cv2.rectangle(img_miss_background, (boxes[idx][0], boxes[idx][1]), (boxes[idx][2], boxes[idx][3]), (255, 0, 0), 1)
                                    #蓝色的gt类比名称
                                    cv2.putText(img_miss_background, cls, (boxes[idx][0], boxes[idx][1] - 2), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)
            else:   # if boxes为假, 没有预测框也没有gt框, 那没事。
                pass
                     
        err_detect_save_path=os.path.join(errDir,'FP(error_detect)')                                   # 设置误检输出路径
        if not os.path.exists(err_detect_save_path):                                                   # 如果没有这个路径，就创建一个
            os.makedirs(err_detect_save_path)
            
        really_err_detect_save_path=os.path.join(err_detect_save_path,'FP_background')                 # 设置真正误检输出路径
        if not os.path.exists(really_err_detect_save_path):                                            # 如果没有这个路径，就创建一个
            os.makedirs(really_err_detect_save_path)
        
        err_detect_other_GT_save_path=os.path.join(err_detect_save_path,'FP_other_GT')                 # 设置误检它类输出路径
        if not os.path.exists(err_detect_other_GT_save_path):                                          # 如果没有这个路径，就创建一个
            os.makedirs(err_detect_other_GT_save_path)
            
        miss_detect_save_path=os.path.join(errDir,'FN(miss_detect)')                                   # 设置漏检输出路径
        if not os.path.exists(miss_detect_save_path):                                                  # 如果没有这个路径，就创建一个
            os.makedirs(miss_detect_save_path)
            
        really_miss_detect_save_path=os.path.join(miss_detect_save_path,'FN_really')                   # 设置真正漏检输出路径
        if not os.path.exists(really_miss_detect_save_path):                                           # 如果没有这个路径，就创建一个
            os.makedirs(really_miss_detect_save_path)
        
        miss_detect_other_predict_save_path=os.path.join(miss_detect_save_path,'FN_other_predict_class') # 设置漏检，却被它类检出的输出路径
        if not os.path.exists(miss_detect_other_predict_save_path):                                    # 如果没有这个路径，就创建一个
            os.makedirs(miss_detect_other_predict_save_path)
        
        if need_auto_save_missing_xml == True:                                                         #这是一个开关，如果你想生成漏标的标签xml, 则判断开关是不是等于True
            really_err_detect_save_path_xml=os.path.join(err_detect_save_path,'really_error_detect_xml') # 设置真正误检xml输出路径
            if not os.path.exists(really_err_detect_save_path_xml):                                      # 如果没有这个路径，就创建一个
                os.makedirs(really_err_detect_save_path_xml)
            really_err_detect_save_path_pic=os.path.join(err_detect_save_path,'really_error_detect_pic') # 设置真正误检pic输出路径
            if not os.path.exists(really_err_detect_save_path_pic):                                      # 如果没有这个路径，就创建一个
                os.makedirs(really_err_detect_save_path_pic)
                
            if len(for_generate_xml) > 0:                                                                #说明当前图片存在误检并且是真实误检,于是转去生成xml预标注
                generate_label(img, for_generate_xml, cls, ori_imgfile,really_err_detect_save_path_xml,really_err_detect_save_path_pic)
          
        if err_flag == True:
            cv_imwrite(os.path.join(err_detect_other_GT_save_path,os.path.splitext(os.path.basename(ori_imgfile))[0] + '.jpg'), img_err_but_have_other_GT)
        if err_flag_really == True:
            if err_flag == True:   #再判断这张图片有没有被写入到other_GT文件夹，如果写入了，这个背景误检文件夹就不写入了。
                pass
            else:
                cv_imwrite(os.path.join(really_err_detect_save_path,os.path.splitext(os.path.basename(ori_imgfile))[0] + '.jpg'), img_err_background)
        if miss_flag_other_predict == True:
            cv_imwrite(os.path.join(miss_detect_other_predict_save_path,os.path.splitext(os.path.basename(ori_imgfile))[0] + '.jpg'), img_miss_with_other_predict)
        if miss_flag == True:
            if miss_flag_other_predict == True: #再判断这张图片有没有被写入到漏检且被其它类别误检文件夹,如果写入了，这个真正漏检文件夹就不写入，避免重复写入。
                pass
            else:
                cv_imwrite(os.path.join(really_miss_detect_save_path,os.path.splitext(os.path.basename(ori_imgfile))[0] + '.jpg'), img_miss_background)
        else:
            pass

          
def select_pic_by_p_r(picpath,ori_xmlpath,detect_txtpath,savepath,classes, iou_thres=0.5,need_auto_save_missing_xml = False,names_path=None):
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    for ori_txtname in os.listdir(detect_txtpath):
        ori_txtfile=os.path.join(detect_txtpath,ori_txtname)
     
        xml_id=os.path.splitext(os.path.basename(ori_txtname))[0]
        #print("xml_id=",xml_id)
        ori_imgfile = get_file_path(picpath,xml_id)
        img = cv_imread(ori_imgfile)
        print(os.path.basename(ori_imgfile))
        ##yolov4
        pre_dict_per = get_txt_result_from_yolov4(ori_txtfile,classes,img)  #如果是yolov4检测处理来的结果，用这行读取，读取预测结果
        ##yolov5、yolov7
        #pre_dict_per = get_txt_result(ori_txtfile,classes,img)     # 如果是yolov5yolov7检测处理来的结果，用这行读取，读取预测结果
        save_err_img(pre_dict_per, ori_imgfile, ori_xmlpath, classes, savepath, iou_thres=iou_thres, need_auto_save_missing_xml = need_auto_save_missing_xml, names_path=names_path)


if __name__=="__main__":
    names_path="F:/VScode/program/analysis_error_miss/jobclothe/classes.names"
    classe_names=get_classes(names_path)
    for i in range(0, len(classe_names)):
        print("开始执行{}类别分析过程".format(classe_names[i]))
        select_pic_by_p_r(
                        picpath="F:/VScode/program/analysis_error_miss/jobclothe/JPEGImages",
                        ori_xmlpath="F:/VScode/program/analysis_error_miss/jobclothe/Annotations",
                        detect_txtpath="F:/VScode/program/analysis_error_miss/jobclothe/labels",
                        savepath="F:/VScode/program/analysis_error_miss/jobclothe/output_"+classe_names[i],
                        classes=[classe_names[i]],
                        iou_thres=0.25,
                        need_auto_save_missing_xml = False,
                        names_path="F:/VScode/program/analysis_error_miss/jobclothe/classes.names"
                        )


