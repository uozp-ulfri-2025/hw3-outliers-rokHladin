from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import os
import math

def embed(fn):
    """Embed the given image with SqueezeNet 1.1.

    Consult https://pytorch.org/hub/pytorch_vision_squeezenet/

    The above link also uses softmax as the final transformation;
    avoid that final step. Convert the output tensor into a numpy
    array and return it.
    """
    model = torch.hub.load("pytorch/vision:v0.10.0", "squeezenet1_1", pretrained=True)

    model.eval()

    input_image = Image.open(fn).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to("cuda")
    #     model.to("cuda")

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes

    return np.array(output[0])


def read_data(path):
    """
    Preberi vse slike v podani poti, jih pretvori v "embedding" in vrni rezultat
    kot slovar, kjer so ključi imena slik, vrednosti pa pripadajoči vektorji.
    """
    slovar = {}
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                relative_path = os.path.join(root, file)
                file_key = os.path.relpath(relative_path, path)
                slovar[file_key] = embed(relative_path)
    return slovar


def euclidean_dist(r1, r2):
    missing = 0
    square_distances = []
    for i, (x, y) in enumerate(zip(r1, r2)):
        if math.isnan(x) or math.isnan(y):
            missing += 1
        else:
            square_distances.append((x - y) ** 2)
    if not square_distances:
        return math.nan
    total_sum = sum(square_distances)
    average_distance = total_sum / len(square_distances)
    total_sum += missing * average_distance
    result = math.sqrt(total_sum)
    return result

def cosine_sim(d1, d2):
    v1 = np.array(d1)
    v2 = np.array(d2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
def cosine_dist(d1, d2):
    """
    Vrni razdaljo med vektorjema d1 in d2, ki je smiselno
    pridobljena iz kosinusne podobnosti.
    """
    return 1 - cosine_sim(d1, d2)


def silhouette(el, clusters, data, distance_fn=euclidean_dist):
    """
    Za element el ob podanih podatkih data (slovar vektorjev) in skupinah
    (seznam seznamov nizov: ključev v slovarju data) vrni silhueto za element el.
    """
    a = 0
    b = float('inf')
    for cluster in clusters:
        avg_distance = 0
        if el in cluster:
            # If element is only one in cluster, silhouette is 0 by definition
            if len(cluster) == 1:
                return 0
            
            a = sum(distance_fn(data[el], data[el2]) for el2 in cluster if el2 != el)
            a /= len(cluster) - 1 if len(cluster) > 1 else 1
            
        else:
            avg_distance = sum(distance_fn(data[el], data[el2]) for el2 in cluster) / len(cluster)
            if avg_distance < b:
                b = avg_distance
    return (b - a) / max(a, b)


def silhouette_average(data, clusters, distance_fn=euclidean_dist):
    """
    Za podane podatke (slovar vektorjev) in skupine (seznam seznamov nizov:
    ključev v slovarju data) vrni povprečno silhueto.
    """
    avg_silhouette = 0
    for el in data:
        avg_silhouette += silhouette(el, clusters, data, distance_fn)
    return avg_silhouette / len(data)


def group_by_dir(names):
    """Generiraj skupine iz direktorijev, v katerih se nahajajo slike"""
    slovar = defaultdict(list)
    for name in names:
        dir_name = os.path.dirname(name)
        slovar[dir_name].append(name)
    return list(slovar.values())


def order_by_decreasing_silhouette(data, clusters):
    cl = []
    for el in data:
        sil = silhouette(el, clusters, data, cosine_dist)
        cl.append((el, sil))
    sorted_list = sorted(cl, key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_list]

if __name__ == "__main__":
    data = read_data("traffic-signs")
    clusters = group_by_dir(data.keys())
    ordered = order_by_decreasing_silhouette(data, clusters)
    print("ATYPICAL TRAFFIC SIGNS")
    for o in ordered[-3:]:
        print(o)
