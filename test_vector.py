import math
import random
import unittest

from vector import Vector


def rms(v):
    return math.sqrt(sum(a * a for a in v) / len(v))


class TestMean(unittest.TestCase):
    def test_known_value(self):
        self.assertAlmostEqual(Vector([1, 2, 3, 4]).mean(), 2.5)

    def test_single_entry(self):
        self.assertAlmostEqual(Vector([7]).mean(), 7.0)

    def test_negative_entries(self):
        self.assertAlmostEqual(Vector([-2, 2, -4, 4]).mean(), 0.0)

    def test_constant_vector(self):
        self.assertAlmostEqual(Vector([5, 5, 5, 5, 5]).mean(), 5.0)

    def test_empty_vector_raises(self):
        with self.assertRaises(ValueError):
            Vector([]).mean()

    def test_mean_is_sum_over_length(self):
        v = Vector([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertAlmostEqual(v.mean(), sum(v) / len(v))

    def test_linearity_addition(self):
        x = Vector([1, 2, 3, 4])
        y = Vector([-3, 0, 5, 2])
        self.assertAlmostEqual((x + y).mean(), x.mean() + y.mean())

    def test_linearity_scalar_multiplication(self):
        x = Vector([1, 2, 3, 4])
        for a in (-3.5, 0, 2, 10):
            self.assertAlmostEqual((a * x).mean(), a * x.mean())

    def test_mean_of_shifted_vector(self):
        x = Vector([1, 2, 3, 4])
        c = 6.5
        shifted = Vector([a + c for a in x])
        self.assertAlmostEqual(shifted.mean(), x.mean() + c)

    def test_mean_equals_dot_with_uniform_weights(self):
        x = Vector([2, 4, 6, 8, 10])
        ones = Vector([1] * len(x))
        self.assertAlmostEqual(x.mean(), ones.dot(x) / len(x))


class TestDemean(unittest.TestCase):
    def test_known_value(self):
        d = Vector([1, 2, 3, 4]).demean()
        expected = [-1.5, -0.5, 0.5, 1.5]
        for a, b in zip(d, expected):
            self.assertAlmostEqual(a, b)

    def test_returns_vector(self):
        self.assertIsInstance(Vector([1, 2, 3]).demean(), Vector)

    def test_preserves_length(self):
        x = Vector([1, 5, 2, 8, 3])
        self.assertEqual(len(x.demean()), len(x))

    def test_original_not_modified(self):
        x = Vector([1, 2, 3, 4])
        before = list(x.entries)
        x.demean()
        self.assertEqual(x.entries, before)

    def test_demeaned_vector_has_zero_mean(self):
        x = Vector([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertAlmostEqual(x.demean().mean(), 0.0)

    def test_demean_is_idempotent(self):
        x = Vector([3, 1, 4, 1, 5, 9, 2, 6])
        once = x.demean()
        twice = once.demean()
        for a, b in zip(once, twice):
            self.assertAlmostEqual(a, b)

    def test_constant_vector_demeans_to_zero(self):
        d = Vector([4, 4, 4, 4]).demean()
        for a in d:
            self.assertAlmostEqual(a, 0.0)

    def test_entries_are_original_minus_mean(self):
        x = Vector([2, 7, 1, 8])
        d = x.demean()
        m = x.mean()
        for xi, di in zip(x, d):
            self.assertAlmostEqual(di, xi - m)

    def test_demean_invariant_to_shift(self):
        x = Vector([1, 2, 3, 4])
        shifted = Vector([a + 100 for a in x])
        for a, b in zip(x.demean(), shifted.demean()):
            self.assertAlmostEqual(a, b)

    def test_demean_scales_with_scalar(self):
        x = Vector([1, 2, 3, 4])
        a = -2.5
        lhs = (a * x).demean()
        rhs = a * x.demean()
        for u, v in zip(lhs, rhs):
            self.assertAlmostEqual(u, v)

    def test_demean_is_orthogonal_to_ones(self):
        x = Vector([3, 1, 4, 1, 5, 9])
        ones = Vector([1] * len(x))
        self.assertAlmostEqual(x.demean().dot(ones), 0.0)


class TestStd(unittest.TestCase):
    def test_known_value(self):
        self.assertAlmostEqual(Vector([1, 2, 3, 4, 5]).std(), math.sqrt(2))

    def test_population_definition(self):
        x = Vector([2, 4, 4, 4, 5, 5, 7, 9])
        self.assertAlmostEqual(x.std(), 2.0)

    def test_constant_vector_has_zero_std(self):
        self.assertAlmostEqual(Vector([3, 3, 3, 3]).std(), 0.0)

    def test_single_entry_has_zero_std(self):
        self.assertAlmostEqual(Vector([42]).std(), 0.0)

    def test_std_is_nonnegative(self):
        rng = random.Random(0)
        for _ in range(20):
            x = Vector([rng.uniform(-10, 10) for _ in range(8)])
            self.assertGreaterEqual(x.std(), 0.0)

    def test_zero_std_implies_constant(self):
        x = Vector([2, 2, 2])
        self.assertAlmostEqual(x.std(), 0.0)
        y = Vector([2, 2, 2.5])
        self.assertGreater(y.std(), 0.0)

    def test_std_equals_rms_of_demeaned(self):
        x = Vector([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertAlmostEqual(x.std(), rms(x.demean()))

    def test_std_invariant_to_shift(self):
        x = Vector([1, 2, 3, 4, 10])
        for c in (-50, 0, 3.3, 1000):
            shifted = Vector([a + c for a in x])
            self.assertAlmostEqual(shifted.std(), x.std())

    def test_std_scales_with_absolute_value_of_scalar(self):
        x = Vector([1, 2, 3, 4, 10])
        for a in (-3, -0.5, 0, 2, 7):
            self.assertAlmostEqual((a * x).std(), abs(a) * x.std())

    def test_std_equals_std_of_demeaned(self):
        x = Vector([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertAlmostEqual(x.std(), x.demean().std())

    def test_rms_std_mean_relationship(self):
        x = Vector([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertAlmostEqual(rms(x) ** 2, x.mean() ** 2 + x.std() ** 2)

    def test_std_not_more_than_rms(self):
        x = Vector([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertLessEqual(x.std(), rms(x) + 1e-12)

    def test_std_equals_rms_when_mean_is_zero(self):
        x = Vector([-2, -1, 0, 1, 2])
        self.assertAlmostEqual(x.mean(), 0.0)
        self.assertAlmostEqual(x.std(), rms(x))

    def test_chebyshev_inequality(self):
        rng = random.Random(1)
        x = Vector([rng.gauss(0, 3) for _ in range(200)])
        m, s = x.mean(), x.std()
        for alpha in (1.5, 2, 3, 4):
            count = sum(1 for a in x if abs(a - m) >= alpha * s)
            self.assertLessEqual(count / len(x), 1 / alpha ** 2 + 1e-12)

    def test_std_of_negated_vector_is_same(self):
        x = Vector([1, 5, 2, 8])
        self.assertAlmostEqual((-x).std(), x.std())


if __name__ == "__main__":
    unittest.main()
