import math

class Rectangle():
    def __init__(self, top_left , bottom_right):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.height = bottom_right.y - top_left.y
        self.width = bottom_right.x - top_left.x
        self.area = self.height * self.width

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def rms (error_list):
    sum = 0
    for i in error_list:
        sum+= math.pow(i,2)
    sum /= len(error_list)
    sum = math.sqrt(sum)
    return sum

def me(error_list):
    if not len(error_list):
        return -1
    sum = 0
    for i in error_list:
        sum+= i
    sum /= len(error_list)
    return sum

def writeInFile(boxes, img_name):
    img_name_file = img_name[:-4]
    file_name = "Annotation/In/generatedbox_"+str(img_name_file)+".txt"
    file = open(file_name,"w")
    for box in boxes:
        width = box[1][0]- box[0][0]
        height = box[1][1] - box[0][1]
        to_write = str(img_name)+", "+str(box[0][0])+", "+str(box[0][1])+", "+str(width)+", "+str(height)+"\n"
        file.write(to_write)
    file.close()

def getRefId(file_path):
    ref_id = []
    text_file = open(file_path, "r")
    lines = text_file.read().split('\n')
    for line in lines :
        list = line.split(',')
        if list[0] == '':
            text_file.close()
            return ref_id
        tmp = list[0][:-4]
        tmp = tmp[-4:]
        t = int(tmp)
        ref_id.append(t)
    text_file.close()
    return ref_id
