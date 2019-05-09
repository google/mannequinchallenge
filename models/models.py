from models import pix2pix_model

def create_model(opt, _isTrain):
    model = None
    #from .pix2pix_model import Pix2PixModel
    model = pix2pix_model.Pix2PixModel(opt, _isTrain)
    print("model [%s] was created" % (model.name()))
    return model
