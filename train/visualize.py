import torch

def color_frame(x, color, lw=1):
    color = torch.tensor(color)
    x[:, :lw, :] = color[:, None, None].expand_as(x[:, :lw, :])
    x[:, :, :lw] = color[:, None, None].expand_as(x[:, :, :lw])
    x[:, -lw:, :] = color[:, None, None].expand_as(x[:, -lw:, :])
    x[:, :, -lw:] = color[:, None, None].expand_as(x[:, :, -lw:])

    return x
