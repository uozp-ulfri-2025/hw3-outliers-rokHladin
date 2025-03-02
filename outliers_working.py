import time
import os
import math
from math import sqrt, isnan
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
from torchvision import transforms


def embed(fn):
    """ Embed the given image with SqueezeNet 1.1.

    Consult https://pytorch.org/hub/pytorch_vision_squeezenet/

    The above link also uses softmax as the final transformation;
    avoid that final step. Convert the output tensor into a numpy
    array and return it.
    """

    model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)
    model.eval()

    # sample execution (requires torchvision)

    input_image = Image.open(fn).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    with torch.no_grad():
        output = model(input_batch)

    return np.array(output[0])


def listimage(path):
    out = []
    fdir = None
    for dir, dirs, files in os.walk(path):
        if fdir is None:
            fdir = dir
        for fn in files:
            if fn.endswith(".jpg") or fn.endswith(".png"):
                cdir = dir.replace(fdir, "").strip("/")
                out.append(os.path.join(cdir, fn))
    out = sorted(out)
    return out


def read_data(path):
    """
    Preberi vse slike v podani poti, jih pretvori v "embedding" in vrni rezultat
    kot slovar, kjer so ključi imena slik, vrednosti pa pripadajoči vektorji.
    """
    imagefn = listimage(path)
    t = time.time()
    data = {name: embed(os.path.join(path, name)) for name in imagefn}
    print("Images converted", time.time() - t)
    return data


def cosine_sim(d1, d2):
    return d1.dot(d2)/(np.linalg.norm(d1)* np.linalg.norm(d2))


def cosine_dist(d1, d2):
    """
    Vrni razdaljo med vektorjema d1 in d2, ki je smiselno
    pridobljena iz kosinusne podobnosti.
    """
    return 1 - cosine_sim(d1, d2)
    cs = cosine_sim(d1, d2)
    angle = math.acos(min(cs, 1))/(math.pi/2)
    return angle

def euclidean_dist(r1, r2):
    res = 0
    count = 0
    for a1, a2 in zip(r1, r2):
        if not isnan(a1) and not isnan(a2):
            res += (a1 - a2) ** 2
            count += 1
    if count == 0 and res == 0:
        return float("nan")
    return sqrt(res / count * len(r1))


def silhouette(el, clusters, data, distance_fn=euclidean_dist):
    """
    Za element el ob podanih podatkih data (slovar vektorjev) in skupinah
    (seznam seznamov nizov: ključev v slovarju data) vrni silhueto za element el.
    """
    a_i = None
    b_i = 10 ** 10
    for cl in clusters:
        if el in cl:
            others = set(cl) - {el}
            if len(others):
                a_i = np.mean([distance_fn(data[el], data[j]) for j in others])
            else:
                return 0.
        else:
            b_cand = np.mean([distance_fn(data[el], data[j]) for j in cl])
            if b_cand < b_i:
                b_i = b_cand
    s_i = (b_i - a_i) / max(a_i, b_i)
    return s_i


def silhouette_average(data, clusters, distance_fn=euclidean_dist):
    """
    Za podane podatke (slovar vektorjev) in skupine (seznam seznamov nizov:
    ključev v slovarju data) vrni povprečno silhueto.
    """
    s = []
    for el in data:
        s.append(silhouette(el, clusters, data, distance_fn=distance_fn))
    return np.mean(s)


def group_by_dir(names):
    """ Generiraj skupine iz direktorijev, v katerih se nahajajo slike """
    g = defaultdict(list)
    for fn in names:
        p = os.path.dirname(fn)
        g[p].append(fn)
    return list(g.values())


def order_by_decreasing_silhouette(data, clusters):
    res = []
    for el in data:
        s = silhouette(el, clusters, data, distance_fn=cosine_dist)
        res.append((s, el))
    res = sorted(res, reverse=True)
    return [n for s, n in res]


if __name__ == "__main__":
    data = read_data("traffic-signs")
    clusters = group_by_dir(data.keys())
    ordered = order_by_decreasing_silhouette(data, clusters)
    print("ATYPICAL TRAFFIC SIGNS")
    for o in ordered[-3:]:
        print(o)