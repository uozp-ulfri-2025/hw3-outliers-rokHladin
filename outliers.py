from sklearn.metrics import euclidean_distances


def embed(fn):
    """ Embed the given image with SqueezeNet 1.1.

    Consult https://pytorch.org/hub/pytorch_vision_squeezenet/

    The above link also uses softmax as the final transformation;
    avoid that final step. Convert the output tensor into a numpy
    array and return it.
    """
    raise NotImplementedError()


def read_data(path):
    """
    Preberi vse slike v podani poti, jih pretvori v "embedding" in vrni rezultat
    kot slovar, kjer so ključi imena slik, vrednosti pa pripadajoči vektorji.
    """
    raise NotImplementedError()


def euclidean_dist(r1, r2):
    raise NotImplementedError()


def cosine_dist(d1, d2):
    """
    Vrni razdaljo med vektorjema d1 in d2, ki je smiselno
    pridobljena iz kosinusne podobnosti.
    """
    raise NotImplementedError()


def silhouette(el, clusters, data, distance_fn=euclidean_dist):
    """
    Za element el ob podanih podatkih data (slovar vektorjev) in skupinah
    (seznam seznamov nizov: ključev v slovarju data) vrni silhueto za element el.
    """
    raise NotImplementedError()


def silhouette_average(data, clusters, distance_fn=euclidean_dist):
    """
    Za podane podatke (slovar vektorjev) in skupine (seznam seznamov nizov:
    ključev v slovarju data) vrni povprečno silhueto.
    """
    raise NotImplementedError()


def group_by_dir(names):
    """ Generiraj skupine iz direktorijev, v katerih se nahajajo slike """
    raise NotImplementedError()


def order_by_decreasing_silhouette(data, clusters):
    raise NotImplementedError()


if __name__ == "__main__":
    data = read_data("traffic-signs")
    clusters = group_by_dir(data.keys())
    ordered = order_by_decreasing_silhouette(data, clusters)
    print("ATYPICAL TRAFFIC SIGNS")
    for o in ordered[-3:]:
        print(o)
