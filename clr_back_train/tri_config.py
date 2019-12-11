import os
CLASSES = ["anti-normal","immune_cells","normal","other_cells"]
MIN_LR = 1e-4
MAX_LR = 1e-1
BATCH_SIZE = 64
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 60
LRFIND_PLOT_PATH = os.path.sep.join(["/cptjack/totem/barrylee/codes/output/all-tri", "99004lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["/cptjack/totem/barrylee/codes/output/all-tri", "99004training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["/cptjack/totem/barrylee/codes/output/all-tri", "99004clr_plot.png"])