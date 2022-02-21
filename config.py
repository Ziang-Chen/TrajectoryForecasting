#---------------------------------------
# Configutation All in One
#---------------------------------------

class conf:
    def __init__(self,setting=None):  # The alternative way to configure by Argparse
        
        self.dataset='./data/'
        self.title="visulization by Ziang Chen @ KCL"
        self.result_base_dir='./'
        self.model='./'


        if not (setting is None):
            self.dataset=setting['dataset']
            self.model=setting['model']
            self.title=setting['title']
            self.result_base_dir=setting['viz_base']
