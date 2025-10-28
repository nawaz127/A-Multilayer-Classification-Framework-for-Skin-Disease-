import argparse, torch
from PIL import Image
from torchvision import transforms
from utils.datasets import LABELS_9
from resnet_model import ResNet50
from DenseNet121 import DenseNet121Medical

def load(model, nc, ckpt, device):
    m = model(nc).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)  # unused but kept for symmetry
    ap.add_argument('--image', required=True)
    ap.add_argument('--l1_ckpt', default='runs/L1/resnet50/best.pt')
    ap.add_argument('--l2_ckpt', default='runs/L2/densenet121/best.pt')
    ap.add_argument('--tau', type=float, default=0.65)
    args=ap.parse_args()

    device='cuda' if torch.cuda.is_available() else 'cpu'
    normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    T = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(), normalize])
    x = T(Image.open(args.image).convert('RGB')).unsqueeze(0).to(device)

    # L1: Normal vs Abnormal
    l1 = load(ResNet50, 2, args.l1_ckpt, device)
    p = l1(x).softmax(1)[0]
    p_normal = p[0].item()

    if p_normal >= args.tau:
        print(f'Prediction: Normal (p={p_normal:.3f})')
    else:
        # L2: 8 diseases (no Normal)
        l2_labels = [c for c in LABELS_9 if c!='Normal']
        l2 = load(DenseNet121Medical, len(l2_labels), args.l2_ckpt, device)
        q = l2(x).softmax(1)[0]
        idx = int(q.argmax().item())
        print(f'Prediction: {l2_labels[idx]} (p={q[idx].item():.3f}; p_normal={p_normal:.3f})')
