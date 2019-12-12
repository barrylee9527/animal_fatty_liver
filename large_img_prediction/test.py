import pickle as pk
from output_process.infiltrating_region_staistics import get_nuclei_classification_result_info
matrix,_= pk.load(open('/cptjack/totem/barrylee/ndpi-maskrcnn/output/output_cc_pickle/#1.pkl','rb'))
nuclei_info=matrix[:4]
result = get_nuclei_classification_result_info(nuclei_info)
print(result)
