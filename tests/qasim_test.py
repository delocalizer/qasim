import unittest
from qasim import qasim

class TestQasim(unittest.TestCase):

    def test_true(self):
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
