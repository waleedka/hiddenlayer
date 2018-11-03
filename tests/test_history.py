import os
import sys
import shutil
import unittest
import hiddenlayer as hl

# Create output directory in project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "test_output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


class TestHistory(unittest.TestCase):
    def test_history(self):
        # Create History object
        h = hl.History()

        for s in range(100):
            loss = (100-s)/100
            accuracy = s / 100
            h.log(s, loss=loss)
            h.log(s, accuracy=accuracy)

        self.assertEqual(h["loss"].data[0], 1)
        self.assertEqual(h["accuracy"].data[0], 0)

        # Save and load
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        h.save(os.path.join(OUTPUT_DIR, "history.pkl"))
        h.history = []
        h.load(os.path.join(OUTPUT_DIR, "history.pkl"))
        self.assertEqual(h["loss"].data[0], 1)
        self.assertEqual(h["accuracy"].data[0], 0)

        # Clean up
        # TODO: shutil.rmtree(OUTPUT_DIR)


if __name__ == "__main__":
    unittest.main()
