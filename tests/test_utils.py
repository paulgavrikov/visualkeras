import unittest
from visualkeras.utils import get_rgba_tuple, self_multiply, get_keys_by_value, fade_color


class UtilMethods(unittest.TestCase):

    def test_get_keys_by_value(self):
        d = {
            'a': 1,
            'aa': 1,
            'b': 2
        }

        self.assertEqual(list(get_keys_by_value(d, 1)), ['a', 'aa'])
        self.assertEqual(list(get_keys_by_value(d, 2)), ['b'])
        self.assertEqual(list(get_keys_by_value(d, 99)), [])

    def test_self_multiply(self):
        x = self_multiply((None, 1, 2, 3))
        self.assertEqual(x, 6)

        x = self_multiply((None,))
        self.assertEqual(x, 0)

        x = self_multiply((44,))
        self.assertEqual(x, 44)

        x = self_multiply((44, None))
        self.assertEqual(x, 44)

        x = self_multiply((0,))
        self.assertEqual(x, 0)

        x = self_multiply((None, 0))
        self.assertEqual(x, 0)

    def test_get_rgba_tuples_by_name(self):
        x = get_rgba_tuple('red')
        y = (255, 0, 0, 255)
        self.assertEqual(x, y)

    def test_get_rgba_tuples_by_str(self):
        x = get_rgba_tuple("rgb(1, 2, 3)")
        y = (1, 2, 3, 255)
        self.assertEqual(x, y)

    def test_get_rgba_tuples_by_hex_str(self):
        x = get_rgba_tuple("#010203")
        y = (1, 2, 3, 255)
        self.assertEqual(x, y)

        x = get_rgba_tuple("#01020304")
        y = (1, 2, 3, 4)
        self.assertEqual(x, y)

    def test_get_rgba_tuples_by_percent(self):
        x = get_rgba_tuple("rgb(100%, 50%, 0%)")
        y = (255, 128, 0, 255)
        self.assertEqual(x, y)

    def test_get_rgba_tuples_by_tuples(self):
        x = get_rgba_tuple((100, 50, 0))
        y = (100, 50, 0, 255)
        self.assertEqual(x, y)

        x = get_rgba_tuple((100, 50, 0, 44))
        y = (100, 50, 0, 44)
        self.assertEqual(x, y)

    def test_get_rgba_tuples_by_int(self):
        x = get_rgba_tuple(0x010203)
        y = (1, 2, 3, 0)
        self.assertEqual(x, y)

        x = get_rgba_tuple(0x01020304)
        y = (2, 3, 4, 1)
        self.assertEqual(x, y)

    def test_fade_color(self):
        self.assertEqual(fade_color((0, 10, 30, 200), 20), (0, 0, 10, 200))


if __name__ == '__main__':
    unittest.main()
