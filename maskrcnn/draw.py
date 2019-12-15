import matplotlib.pyplot as plt
import os
import configparser
from output_process.infiltrating_region_staistics import get_nuclei_amount_matrix,get_nuclei_amount_region_matrix
from output_process.cancer_heatmap import create_heatmap
current_path = os.path.dirname(__file__)
conf = configparser.ConfigParser()
conf.read(os.path.join(current_path, ".", "sys.ini"))
svs_file = conf.get("UTILS_HEATMAP", "SVS_DIR")
pkl_file = conf.get("UTILS_HEATMAP", "PKL_DIR")
print(svs_file)
step = 100
out_dir = '/cptjack/totem/barrylee/codes/output/heatmap+++'
for i in os.listdir(svs_file):
    prefix_name = i.split('.')[0]
    svs_name = svs_file+'/'+i
    pkl_name = pkl_file+'/'+prefix_name+'.pkl'
    print(prefix_name)
    if prefix_name == '#45':
        print(svs_name, pkl_name)
        immune_cells_nuclei_matrix, other_cells_nuclei_matrix, \
        ballooning_nuclei_matrix, normal_nuclei_matrix, \
        steatosis_nuclei_matrix = get_nuclei_amount_region_matrix(pkl_name, svs_name, step)
        # print(immune_cells_nuclei_matrix, other_cells_nuclei_matrix,ballooning_nuclei_matrix, normal_nuclei_matrix, steatosis_nuclei_matrix)
        create_heatmap(svs_name, immune_cells_nuclei_matrix, prefix_name, step, 'immune_cells', out_dir)
        create_heatmap(svs_name, other_cells_nuclei_matrix, prefix_name, step, 'other_cells', out_dir)
        create_heatmap(svs_name, ballooning_nuclei_matrix, prefix_name, step, 'ballooning_cells', out_dir)
        create_heatmap(svs_name, normal_nuclei_matrix, prefix_name, step, 'normal_cells', out_dir)
        create_heatmap(svs_name, steatosis_nuclei_matrix, prefix_name, step, 'steatosis_cells', out_dir)
    else:
        print('no found file')

