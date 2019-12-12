# @Time    : 2018.10.29
# @Author  : kawa Yeung
# @Licence : bio-totem


import os
import glob

from utils.openslide_utils import Slide


def svs_region_patch(svs_file, patch_size):
    """
    Get svs patch size region, and put the inference information to it
    :param svs_file: svs file
    :param patch_size: tuple (width, height), patch size
    :return:
    """

    slide = Slide(svs_file)
    svs_width, svs_height = slide.get_level_dimension(0)
    locations = [(0.35, 0.35), (0.50, 0.35), (0.7, 0.35), \
                 (0.40, 0.50), (0.50, 0.50), (0.65, 0.50), \
                 (0.40, 0.65), (0.50, 0.65), (0.7, 0.7), (0.60, 0.60)]
    locations = [tuple(map(int, (w*svs_width, h*svs_height))) for w, h in locations]

    region_ims = []
    for ii, location in enumerate(locations):
        im = slide.read_region(location, 0, patch_size)
        im.save(os.path.join(output_path, f"{slide.get_basename()}_R{ii}.png"))
        region_ims.append(im)

    return region_ims


if __name__ == "__main__":
    current_path = os.path.dirname(__file__)
    output_path = os.path.join(current_path, "..", "output", "output_region_path")
    if not os.path.isdir(output_path): os.makedirs(output_path)

    svs_path = None # svs file path
    svs_files = glob.glob(os.path.join(svs_path, "*.svs"))

    for svs in svs_files:
        svs_region_patch(svs, (1024, 1024))


