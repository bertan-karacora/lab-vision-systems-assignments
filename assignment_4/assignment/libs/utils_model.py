def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True
    return model
