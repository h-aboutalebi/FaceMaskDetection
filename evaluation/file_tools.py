import sys
from pathlib import Path
import os
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def find_image_files(path,extension):
    image_files=[]
    image_files_name=[]
    for path in Path(path).rglob('*.'+extension):
        image_files.append(path)
        image_files_name.append(path.name)
    return image_files, image_files_name


def find_xml_files(path):
    xml_files = []
    xml_files_name = []
    for path in Path(path).rglob('*.xml'):
        xml_files.append(path)
        xml_files_name.append(path.name)
    return xml_files, xml_files_name

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text #"face" refers to without mask and "face_mask" refers to mask
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects

