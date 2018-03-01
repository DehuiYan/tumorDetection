#!/usr/bin/env python
# coding:utf-8

from lxml import etree
import pprint

def parsexml(xmlname):
    xml_file = etree.parse(xmlname)
    root = xml_file.getroot()
    bbox = []
    for i in root.findall('Annotations'):
        for j in i.findall('Annotation'):
            if j.get('PartOfGroup') == '_0' or j.get('PartOfGroup') == '_1' or j.get('PartOfGroup') == 'metastases':
                for k in j.findall('Coordinates'):
                    xmax = 0
                    xmin = 999999
                    ymax = 0
                    ymin = 999999
                    for l in k.findall('Coordinate'):
                        x = l.get('X')
                        y = l.get('Y')
                        x = float(x)
			x = int(x)
                        y = float(y)
			y = int(y)
                        if x > xmax:
                            xmax = x
                        if x < xmin:
                            xmin = x
                        if y > ymax:
                            ymax = y
                        if y < ymin:
                            ymin = y
                    bbox.append([xmin, ymin, xmax-xmin, ymax-ymin])
    return bbox



def setimginfo(node_root, filename, shape):
    #node_root = etree.Element('annotation')
    node_folder = etree.SubElement(node_root, 'folder')
    node_folder.text = 'VOC2007'

    node_filename = etree.SubElement(node_root, 'filename')
    node_filename.text = filename

    node_size = etree.SubElement(node_root, 'size')
    node_width = etree.SubElement(node_size, 'width')
    node_width.text = str(shape[1])

    node_height = etree.SubElement(node_size, 'height')
    node_height.text = str(shape[0])

    node_depth = etree.SubElement(node_size, 'depth')
    node_depth.text = str(shape[2])

    node_segmented = etree.SubElement(node_root, 'segmented')
    node_segmented.text = '0'

def setobjinfo(node_root, xmin, ymin, xmax, ymax, area, extent, variety):
    node_object = etree.SubElement(node_root, 'object')
    node_name = etree.SubElement(node_object, 'name')
    node_name.text = variety
    node_pose = etree.SubElement(node_object, 'pose')
    node_pose.text = 'Unspecified'
    node_truncated = etree.SubElement(node_object, 'truncated')
    node_truncated.text = '0'
    node_difficult = etree.SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    node_bndbox = etree.SubElement(node_object, 'bndbox')
    node_xmin = etree.SubElement(node_bndbox, 'xmin')
    node_xmin.text = xmin
    node_ymin = etree.SubElement(node_bndbox, 'ymin')
    node_ymin.text = ymin
    node_xmax = etree.SubElement(node_bndbox, 'xmax')
    node_xmax.text = xmax
    node_ymax = etree.SubElement(node_bndbox, 'ymax')
    node_ymax.text = ymax
    node_area = etree.SubElement(node_object, 'area')
    node_area.text = area
    node_extent = etree.SubElement(node_object, 'extent')
    node_extent.text = extent

def makexml(outputXmlDir, number, shape, bbox):
    node_root = etree.Element('annotation')
    c = str(number)
    ze = 6 - len(c)
    filename = '0'*ze + c + '.jpg'
    xmlname = outputXmlDir+'0'*ze + c + '.xml'
    setimginfo(node_root, filename, shape)
    for i in bbox:
    	setobjinfo(node_root, str(i[0]), str(i[1]), str(i[2]), str(i[3]), str(i[4]), str(i[5]), str(i[6]))
    xml = etree.tostring(node_root, pretty_print=True)  #格式化显示，该换行的换行
    with open(xmlname,'w') as f:
    	f.write(xml)
    #dom = parseString(xml)
    #print xml
if __name__ == "__main__":
    bbox = ((23,12,45,65, 200, 0.5, 'ITC'),(12,34,78,57,300,0.8, 'Micro'))
    makexml('', 666, (512, 512, 3), bbox)
    #print parsexml('Tumor_058.xml')
