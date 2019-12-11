# @Time    : 2018.10.18
# @Author  : kawa Yeung
# @Licence : bio-totem


import os
import shutil
import lxml.etree as ET

from PIL import ImageDraw
import numpy as np

current_path = os.path.dirname(__file__)


def xml_to_region(xml_file):
    """
    parse XML label file and get the points
    :param xml_file: xml file
    :return: region list
    """

    tree = ET.parse(xml_file)
    region_list = []
    for region in tree.findall('.//Annotation/Regions/Region'):
        vertex_list = []
        for vertex in region.findall('.//Vertices/Vertex'):
            # parse the 'X' and 'Y' for the vertex
            vertex_list.append(vertex.attrib)
        region_list.append(vertex_list)

    return region_list


def region_handler(im, regions, level_downsample):
    """
    handle region label point to discrete point, and draw the region point to line
    :param im: the image painted in region line
    :param regions: region list, region point,
                    eg : [[{'X': '27381.168113', 'Y': '37358.653791'}], [{'X': '27381.168113', 'Y': '37358.653791'}]]
    :param level_downsample: slide level down sample
    :return: image painted in region line of numpy array format
    """

    region_list = []
    dr = ImageDraw.Draw(im)
    for _, region in enumerate(regions):
        point_list = []
        for __, point in enumerate(region):
            X, Y = int(float(point['X'])/level_downsample), int(float(point['Y'])/level_downsample)
            point_list.append((X, Y))

        region_list.append(point_list)

        points_length = len(point_list)
        x_max = max(point_list, key=lambda point: point[0])[0]
        x_min = min(point_list, key=lambda point: point[0])[0]
        y_max = max(point_list, key=lambda point: point[1])[1]
        y_min = min(point_list, key=lambda point: point[1])[1]
        # mislabeled, here checked by x and y coordinate max and min difference
        if (x_max - x_min < 50) or (y_max - y_min < 50): continue
        if points_length == 2:
            dr.arc(point_list, 0, 360, fill='#ff0000', width=8)
        else:
            dr.line(point_list, fill="#ff0000", width=5)

    return np.asarray(im)


class Region:
    """"
    handle the template xml format file to insert label svs region
    """
    def __init__(self, xml_file):
        parser = ET.XMLParser(remove_blank_text=True)
        if not os.path.isfile(xml_file):
            template = os.path.join(current_path, "template.xml")
            shutil.copy(template, xml_file)
        self._xml_file = xml_file
        self._tree = ET.parse(xml_file, parser)

    def get_region(self, region_id):
        """
        get region by region id
        :param region_id: region id, 0: green, 1: yellow, 2: red, see the template.xml
        :return: the region
        """

        return self._tree.findall(".//Annotation/Regions")[region_id]

    def add(self, region_id, points):
        """
        add one region to the specified region by region id, the added region is ellipse
        and the parameter points is a rectangle bounded by an ellipse
        :param points: list with two element(upper-left, bottom-right), is the rectangle bounded by an ellipse
        :return:
        """

        region = self.get_region(region_id)
        region_num = len(region.findall(".//Region"))
        region_attr = {
            "Id": str(region_num+1),
            "Type": "2",
            "Zoom": "1",
            "Selected": "1",
            "ImageLocation": "",
            "ImageFocus": "0",
            "Length": "80",
            "Area": "400",
            "LengthMicrons": "20",
            "AreaMicrons": "30",
            "Text": "",
            "NegativeROA": "0",
            "InputRegionId": "0",
            "Analyze": "0",
            "DisplayId": "1"
        }

        region_tag = ET.Element("Region", region_attr)
        region.append(region_tag)

        attributes = ET.SubElement(region_tag, "Attributes")
        vertices = ET.Element("Vertices")
        region_tag.append(vertices)

        for point in points:
            # insert point
            ET.SubElement(vertices, "Vertex", attrib=point)

    def save(self):
        """
        save the xml file
        :return:
        """

        self._tree.write(self._xml_file, pretty_print=True)
