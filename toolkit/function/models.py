from ModelCreated.ViTASIGEV.ViTASIGEV import ViTASIGEVModel

def prepare_model(hparams,specific_model=None):
    if hparams.network == 'ViTASIGEV':
        model = load_ViTASIGEV_model(hparams)
    return model

def load_ViTASIGEV_model(hparams): 
    model = ViTASIGEVModel(hparams.ViTAS_dic)
    return model
