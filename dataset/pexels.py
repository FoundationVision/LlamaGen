from torchvision.datasets import ImageFolder

def build_pexels(args, transform):
    return ImageFolder(args.data_path, transform=transform)