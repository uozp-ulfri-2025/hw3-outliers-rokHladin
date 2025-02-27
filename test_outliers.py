import os
import shutil
import tempfile
import unittest
from math import sqrt

import numpy as np
from PIL import Image


from outliers import embed, read_data, cosine_dist, euclidean_dist, \
    silhouette, silhouette_average, group_by_dir, order_by_decreasing_silhouette


class EmbedTest(unittest.TestCase):

    def test_embed(self):
        tempdir = tempfile.mkdtemp()
        try:
            data = np.zeros((10, 10, 3), dtype=np.uint8)
            data[0:3, 0:3] = [255, 0, 0]
            img = Image.fromarray(data, 'RGB')
            fn = os.path.join(tempdir, 'image1.png')
            img.save(fn)
            vec = embed(fn)
            self.assertIsInstance(vec, np.ndarray)
            self.assertEqual(vec.shape, (1000,))
            np.testing.assert_almost_equal(
                vec[:4],
                [2.0275, 3.8617, 2.3489, 2.442],
                decimal=1)
        finally:
            # remove the temp images
            shutil.rmtree(tempdir)

    def test_embed_bw(self):
        # SqueezeNet needs RGB images on input, convert accordingly
        tempdir = tempfile.mkdtemp()
        try:
            data = np.zeros((10, 10), dtype=np.uint8)
            data[0:3, 0:3] = 255
            img = Image.fromarray(data)
            fn = os.path.join(tempdir, 'image1.png')
            img.save(fn)
            vec = embed(fn)
            self.assertIsInstance(vec, np.ndarray)
            self.assertEqual(vec.shape, (1000,))
            np.testing.assert_almost_equal(
                vec[:4],
                [1.3904, 2.7239, 2.4826, 2.4455],
                decimal=1)
        finally:
            # remove the temp images
            shutil.rmtree(tempdir)

    def test_embed_problematic(self):
        # SqueezeNet needs RGB images on input, convert accordingly
        vec = embed('traffic-signs/regulatory/Bike (left) and Pedestrian Lane.png')
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.shape, (1000,))
        np.testing.assert_almost_equal(
            vec[:4],
            [2.576, 4.208, 7.064, 5.167],
            decimal=1)

    def test_read_data(self):
        tempdir = tempfile.mkdtemp()
        try:
            data = np.zeros((10, 10, 3), dtype=np.uint8)
            data[0:3, 0:3] = [255, 0, 0]
            img = Image.fromarray(data, 'RGB')
            img.save(os.path.join(tempdir, 'image1.png'))
            img.save(os.path.join(tempdir, 'image2.jpg'))
            subdir = os.path.join(tempdir, "sub")
            os.mkdir(subdir)
            img.save(os.path.join(subdir, 'image3.jpg'))
            data = read_data(tempdir)
            self.assertIsInstance(data, dict)
            self.assertEqual(len(data), 3)
            self.assertIn('image1.png', data)
            self.assertIn('image2.jpg', data)
            self.assertIn('sub/image3.jpg', data)
            for v in data.values():
                self.assertIsInstance(v, np.ndarray)
                self.assertEqual(v.shape, (1000,))
        finally:
            # remove the temp images
            shutil.rmtree(tempdir)

    def test_read_data_signs(self):
        data = read_data("traffic-signs")
        self.assertIn('regulatory/Bike (left) and Pedestrian Lane.png', data)
        self.assertIn('warning/Peage.jpg', data)
        self.assertEqual(len(data), 70)  # there are 70 images in the data set


dataS4 = {"X": [1, 1],
          "Y": [0.9, 1],
          "Z": [1, 0],
          "Z1": [0.8, 0],
          "A": [0.5, 0.5],
          "A1": [0.6, 0.7],
          "A2": [0.4, 0.3],
          "A3": [0.8, 0.7],
          "B": [0.2, 0.1],
          "B2": [0.4, 0.5],
          }


dataS2 = {"X": [1, 1],
          "Y": [0.9, 1],
          "Z": [1, 0],
          "Z1": [0.8, 0]}


def to_numpy(idict):
    od = {}
    for n, c in idict.items():
        od[n] = np.array(c)
    return od


dataS4 = to_numpy(dataS4)
dataS2 = to_numpy(dataS2)


class SilhouetteTestEuclidean(unittest.TestCase):

    def test_euclidean_distance(self):
        dist = euclidean_dist(np.array([1,2,3]), np.array([1,3,5]))
        self.assertAlmostEqual(dist, sqrt(0+1+4))
        dist = euclidean_dist(np.array([4,2,3]), np.array([1,3,5]))
        self.assertAlmostEqual(dist, sqrt(9+1+4))

    def test_silhouette_basic(self):
        data = {"X": [1, 1],
                "Y": [0.9, 1],
                "Z": [1, 0]}
        data = to_numpy(data)

        s1 = silhouette("X", [["X", "Y"], ["Z"]], data)
        self.assertTrue(0.5 < s1 < 1)

    def test_silhouette_average_basic(self):
        data = {"X": [1, 1],
                "Y": [0.9, 1],
                "Z": [1, 0]}
        data = to_numpy(data)

        s1 = silhouette_average(data, [["X", "Y"], ["Z"]])  # boljše skupine
        s2 = silhouette_average(data, [["X", "Z"], ["Y"]])  # slabše skupine
        s3 = silhouette_average(data, [["Y", "Z"], ["X"]])  # še slabše skupine
        self.assertLess(s2, s1)
        self.assertLess(s3, s2)

    def test_silhouette1(self):
        data = dataS2
        self.assertAlmostEqual(silhouette("X", [["X"], ["Z", "Z1"]], data), 0.0)  # 0 by definition
        self.assertAlmostEqual(silhouette("Z", [["X"], ["Z", "Z1"]], data), 0.8, places=5)

    def test_silhouette2_mean(self):
        data = dataS2
        s1 = silhouette_average(data, [["X", "Y"], ["Z", "Z1"]])  # boljše skupine
        s2 = silhouette_average(data, [["X", "Z"], ["Y", "Z1"]])  # slabše skupine
        s3 = silhouette_average(data, [["Y", "Z"], ["X", "Z1"]])  # še slabše skupine
        self.assertAlmostEqual(s1, 0.851105768704104, places=5)
        self.assertAlmostEqual(s2, -0.42024432102251297, places=5)
        self.assertAlmostEqual(s3, -0.43077817395149687, places=5)

    def test_silhouette2(self):
        data = dataS2
        self.assertAlmostEqual(silhouette("X", [["X", "Y"], ["Z", "Z1"]], data), 0.9009804864072152, places=5)
        self.assertAlmostEqual(silhouette("Y", [["X", "Y"], ["Z", "Z1"]], data), 0.9004962809790011, places=5)
        self.assertAlmostEqual(silhouette("Z", [["X", "Y"], ["Z", "Z1"]], data), 0.800497515516439, places=5)
        self.assertAlmostEqual(silhouette("Z1", [["X", "Y"], ["Z", "Z1"]], data), 0.8024487919137608, places=5)

        self.assertAlmostEqual(silhouette("X", [["X", "Z"], ["Y", "Z1"]], data), -0.44009804864072155, places=5)
        self.assertAlmostEqual(silhouette("Y", [["X", "Z"], ["Y", "Z1"]], data), -0.4502481404895006, places=5)
        self.assertAlmostEqual(silhouette("Z", [["X", "Z"], ["Y", "Z1"]], data), -0.39750621894395555, places=5)
        self.assertAlmostEqual(silhouette("Z1", [["X", "Z"], ["Y", "Z1"]], data), -0.39312487601587404, places=5)

    def test_silhouette4(self):
        data = dataS4
        cl1 = [["X", "Y"], ["Z", "Z1"], ["A", "A1", "A2", "A3"], ["B", "B2"]]
        cl2 = [["Z", "Z1"], ["A", "A1", "A2", "A3"], ["X", "Y"], ["B", "B2"]]
        self.assertAlmostEqual(silhouette("X", cl1, data), 0.8393326749789924, places=5)
        self.assertAlmostEqual(silhouette("Y", cl1, data), 0.8215111850702933, places=5)
        self.assertAlmostEqual(silhouette("A", cl1, data), 0.10247919661515911, places=5)
        self.assertAlmostEqual(silhouette("B2", cl1, data), -0.42418101867909697, places=5)
        self.assertAlmostEqual(silhouette("X", cl2, data), 0.8393326749789924, places=5)
        self.assertAlmostEqual(silhouette("Y", cl2, data), 0.8215111850702933, places=5)
        self.assertAlmostEqual(silhouette("A", cl2, data), 0.10247919661515911, places=5)
        self.assertAlmostEqual(silhouette("B2", cl2, data), -0.42418101867909697, places=5)


class SilhouetteTestCosine(unittest.TestCase):

    def test_cosine_dist(self):
        d1 = np.array([1, 1, 0, 0])
        d2 = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(cosine_dist(d1, d2), 1)
        d1 = np.array([1, 1, 0, 0])
        d2 = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(cosine_dist(d1, d2), 0)

    def test_cosine_dist2(self):
        d1 = np.array([1, 2])
        d2 = np.array([2, 2])
        self.assertAlmostEqual(cosine_dist(d1, d2), 0.05131670194948623, places=4)
        d1 = np.array([1, 2, 0])
        d2 = np.array([2, 2, 3])
        self.assertAlmostEqual(cosine_dist(d1, d2), 0.3492086265440315, places=4)

    def test_silhouette_basic(self):
        data = {"X": np.array([1, 1]),
                "Y": np.array([0.9, 1]),
                "Z": np.array([1, 0])}

        s1 = silhouette("X", [["X", "Y"], ["Z"]], data)
        self.assertTrue(0.5 < s1 < 1)

    def test_silhouette_average_basic(self):
        data = {"X": np.array([1, 1]),
                "Y": np.array([0.9, 1]),
                "Z": np.array([1, 0])}

        s1 = silhouette_average(data, [["X", "Y"], ["Z"]], distance_fn=cosine_dist)  # boljše skupine
        s2 = silhouette_average(data, [["X", "Z"], ["Y"]], distance_fn=cosine_dist)  # slabše skupine
        s3 = silhouette_average(data, [["Y", "Z"], ["X"]], distance_fn=cosine_dist)  # še slabše skupine
        self.assertLess(s2, s1)
        self.assertLess(s3, s2)

    def test_silhouette1(self):
        data = dataS2
        self.assertAlmostEqual(silhouette("X", [["X"], ["Z", "Z1"]], data, distance_fn=cosine_dist), 0.0)  # 0 by definition
        self.assertAlmostEqual(silhouette("Z", [["X"], ["Z", "Z1"]], data, distance_fn=cosine_dist), 1.0, places=5)

    def test_silhouette2_mean(self):
        data = dataS2
        s1 = silhouette_average(data, [["X", "Y"], ["Z", "Z1"]], distance_fn=cosine_dist)  # boljše skupine
        s2 = silhouette_average(data, [["X", "Z"], ["Y", "Z1"]], distance_fn=cosine_dist)  # slabše skupine
        s3 = silhouette_average(data, [["Y", "Z"], ["X", "Z1"]], distance_fn=cosine_dist)  # še slabše skupine
        self.assertAlmostEqual(s1, 0.997776419211436, places=5)
        self.assertAlmostEqual(s2, -0.4970126344010164, places=5)
        self.assertAlmostEqual(s3, -0.49701263440101656, places=5)

    def test_silhouette2(self):
        data = dataS2
        self.assertAlmostEqual(silhouette("X", [["X", "Y"], ["Z", "Z1"]], data, distance_fn=cosine_dist), 0.9952809741615398, places=5)
        self.assertAlmostEqual(silhouette("Y", [["X", "Y"], ["Z", "Z1"]], data, distance_fn=cosine_dist), 0.9958247026842044, places=5)
        self.assertAlmostEqual(silhouette("Z", [["X", "Y"], ["Z", "Z1"]], data, distance_fn=cosine_dist), 1.0, places=5)
        self.assertAlmostEqual(silhouette("Z1", [["X", "Y"], ["Z", "Z1"]], data, distance_fn=cosine_dist), 1.0, places=5)

        self.assertAlmostEqual(silhouette("X", [["X", "Z"], ["Y", "Z1"]], data, distance_fn=cosine_dist), -0.4976404870807699, places=5)
        self.assertAlmostEqual(silhouette("Y", [["X", "Z"], ["Y", "Z1"]], data, distance_fn=cosine_dist), -0.49791235134210204, places=5)
        self.assertAlmostEqual(silhouette("Z", [["X", "Z"], ["Y", "Z1"]], data, distance_fn=cosine_dist), -0.4348874485407752, places=5)
        self.assertAlmostEqual(silhouette("Z1", [["X", "Z"], ["Y", "Z1"]], data, distance_fn=cosine_dist), -0.5576102506404185, places=5)

    def test_silhouette4(self):
        data = dataS4
        cl1 = [["X", "Y"], ["Z", "Z1"], ["A", "A1", "A2", "A3"], ["B", "B2"]]
        cl2 = [["Z", "Z1"], ["A", "A1", "A2", "A3"], ["X", "Y"], ["B", "B2"]]
        self.assertAlmostEqual(silhouette("X", cl1, data, distance_fn=cosine_dist), 0.6365306375452028, places=5)
        self.assertAlmostEqual(silhouette("Y", cl1, data, distance_fn=cosine_dist), 0.7998054270975221, places=5)
        self.assertAlmostEqual(silhouette("A", cl1, data, distance_fn=cosine_dist), -0.8636989890794272, places=5)
        self.assertAlmostEqual(silhouette("B2", cl1, data, distance_fn=cosine_dist), -0.9576160187276127, places=5)
        self.assertAlmostEqual(silhouette("X", cl2, data, distance_fn=cosine_dist), 0.6365306375452028, places=5)
        self.assertAlmostEqual(silhouette("Y", cl2, data, distance_fn=cosine_dist), 0.7998054270975221, places=5)
        self.assertAlmostEqual(silhouette("A", cl2, data, distance_fn=cosine_dist), -0.8636989890794272, places=5)
        self.assertAlmostEqual(silhouette("B2", cl2, data, distance_fn=cosine_dist), -0.9576160187276127, places=5)


class TestFinal(unittest.TestCase):

    def test_group_by_dir(self):
        r = group_by_dir(["a/1", "a/2", "b/3", "a/4"])
        g1 = ["a/1", "a/2", "a/4"]
        g2 = ["b/3"]
        self.assertEqual(len(r), 2)
        self.assertIn(g1, r)
        self.assertIn(g2, r)

    def test_order_by_decreasing_silhouette(self):
        order = order_by_decreasing_silhouette(dataS2, [["X", "Y"], ["Z", "Z1"]])
        self.assertEqual(order[-2:], ['Y', 'X'])

    def test_unusual_traffic_signs(self):
        data = read_data("traffic-signs")
        clusters = group_by_dir(data.keys())
        ordered = order_by_decreasing_silhouette(data, clusters)
        atypical = list(reversed(ordered[-3:]))
        for n in atypical:
            self.assertTrue(n.startswith("warning"))
        self.assertIn("W", atypical[0])
        self.assertIn("op", atypical[1])
        self.assertIn("No", atypical[2])
        print("UNUSUAL SIGNS", atypical)


if __name__ == "__main__":
    unittest.main(verbosity=2)
