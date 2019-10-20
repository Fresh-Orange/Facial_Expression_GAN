
def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'PATN':
        #assert opt.dataset_mode == 'keypoint'
        from .PATN import TransferModel
        model = TransferModel()
    elif opt.model == 'BLEND':
        assert opt.dataset_mode == 'blend_keypoint'
        from .Blend import BlendModel
        model = BlendModel()

    elif opt.model == 'BLEND2':
        #assert opt.dataset_mode == 'blend_keypoint'
        from .Blend2 import BlendModel
        model = BlendModel()

    elif opt.model == 'PATNRES':
        from .PATNRES import TransferModel
        model = TransferModel()

    elif opt.model == 'PATNSN':
        from .PATNSN import TransferModel
        model = TransferModel()

    elif opt.model == 'BLENDSN':
        from .BlendSN import BlendModel
        model = BlendModel()
    elif opt.model == 'BLENDSN_CHANGE':
        from .BlendSN_change import BlendModel
        model = BlendModel()

    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
