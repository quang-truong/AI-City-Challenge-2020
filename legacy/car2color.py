import glob
import re
import os.path as osp 
import xml.etree.ElementTree as ET
import os 
import re
import pickle

def read_json_old(filename,dataset):
    imgpath = []
    xmlp = ET.XMLParser(encoding = "utf-8")
    tree = ET.parse(filename,parser=xmlp)
    root = tree.getroot()
    for elem in root:
        for subelem in elem:
            vehicleid = subelem.attrib['vehicleID']
            imagename = subelem.attrib['imageName']
            imgpath.append((vehicleid,imagename))
    newimgpath = []
    for image_name in dataset: 
        for vehicle_id,imagename in imgpath:
            if imagename == image_name:
                newimgpath.append((vehicle_id,image_name))
    newimgpath = sorted(newimgpath)
    unique_id = []
    exist_id = []
    for vehicle_id, image_name in newimgpath:
        if vehicle_id in exist_id:
            pass
        else: 
            exist_id.append(vehicle_id)
            unique_id.append(image_name)
    return imgpath, unique_id
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval
def natural_keys(text):
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]
def write_log(log_file,imgname): 
    f = open(log_file,"a")
    f.write(imgname)
    f.close()
def read_log_file(log_file):
    f = open(log_file,"r")
    lines = f.readlines()
    count = 0
    lines_array =[]
    cartype = []
    for line in lines:
        line = line.rstrip()
        count+=1
        lines_array.append(line)
        img_name = line.split(' ',1)[0]
        car_type = line.split(' ',1)[1]
        cartype.append((img_name,car_type))
    return cartype
def mapping(id_set, cartype_set):
    cartypeid = []
    for vehicle_id, imagename in id_set:
        for image_cartype, cartype in cartype_set:
            if(image_cartype == imagename):
                cartypeid.append((vehicle_id,cartype))
                pass
    return cartypeid
def mapping_all_id(cartypeid,imgpath):
    final_list = []
    cardict = dict()
    for vehicle_id_car, cartype in cartypeid:
        for vehicle_id,imagename in imgpath:
            if vehicle_id_car == vehicle_id:
                #list_cartype.append(int(cartype))
                newkey = {imagename:int(cartype)}
                cardict.update(newkey)
                #list_img.append(imagename)
                final_list.append((imagename,vehicle_id,cartype))
    #zipobj = zip(list_img,list_cartype)
    #dictofcartype = dict(zipobj)
    return cardict
def categorize_cartype():
    file = open('train_track.txt','r')
    Lines = file.readlines()
    count = 0
    lines_array = []
    list_img = []
    for line in Lines:
        line = line.rstrip()
        count += 1
        lines_array.append(line)
        character = line.split(' ', 1)[0]
        first_img = str(character)
        list_img.append(first_img)
        list_img.sort(key=natural_keys)
    for imgname in list_img:
        write_log("otherlog.txt", imgname +"\n")
    imgpath, set_data = read_json_old("train_label.xml",list_img)
    set_data.sort(key=natural_keys)
    cartype = read_log_file('log_file_unique_final.txt')
    cartypeid = mapping(imgpath,cartype)
    dictionary = mapping_all_id(cartypeid,imgpath)
    print(len(dictionary))
    return dictionary
if __name__ == "__main__":
    dictionary = categorize_cartype()
    #print(dictionary)
    pickle_out = open("car2color.pickle","wb")
    pickle.dump(dictionary,pickle_out)
    pickle_out.close()
    pickle_in = open("car2color.pickle","rb")
    example_dict = pickle.load(pickle_in)
       
    

