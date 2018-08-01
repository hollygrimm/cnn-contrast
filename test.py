import tensorflow as tf
import main
from tensorflow.python.platform import gfile

class MainTest(tf.test.TestCase):
    def test_default(self):
        self.assertEqual(1, 1)

if __name__ == '__main__':
    tf.test.main()
