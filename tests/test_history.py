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
    def test_steps(self):
        # Create History object
        h = hl.History()

        for s in range(100):
            loss = (100-s)/100
            accuracy = s / 100
            h.log(s, loss=loss)
            h.log(s, accuracy=accuracy)

        self.assertEqual(h["loss"].data[0], 1)
        self.assertEqual(h["accuracy"].data[0], 0)
        self.assertEqual(h.metrics, {"loss", "accuracy"})

        # Save and load
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        h.save(os.path.join(OUTPUT_DIR, "history.pkl"))
        
        # Load it
        h2 = hl.History()
        h2.load(os.path.join(OUTPUT_DIR, "history.pkl"))
        self.assertEqual(h["loss"].data[0], h2["loss"].data[0])
        self.assertEqual(h["accuracy"].data[0], h2["accuracy"].data[0])
        self.assertEqual(h2.step, 99)
        self.assertEqual(h2.metrics, {"loss", "accuracy"})
        self.assertEqual(hl.history.format_step(h2.step), "99")
        self.assertEqual(hl.history.format_step(h2.step, zero_prefix=True), "000099")

        # Clean up
        shutil.rmtree(OUTPUT_DIR)

    def test_epochs(self):
        # Create History object
        h = hl.History()

        for e in range(10):
            for s in range(100):
                loss = (100-s)/100
                accuracy = s / 100
                h.log((e, s), loss=loss)
                h.log((e, s), accuracy=accuracy)

        self.assertEqual(h["loss"].data[0], 1)
        self.assertEqual(h["accuracy"].data[0], 0)

        # Save and load
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        h.save(os.path.join(OUTPUT_DIR, "history_epoch.pkl"))
        
        # Load it
        h2 = hl.History()
        h2.load(os.path.join(OUTPUT_DIR, "history_epoch.pkl"))
        self.assertEqual(h["loss"].data[0], h2["loss"].data[0])
        self.assertEqual(h["accuracy"].data[0], h2["accuracy"].data[0])
        self.assertEqual(h2.step, (9, 99))
        self.assertEqual(h2.metrics, {"loss", "accuracy"})
        self.assertEqual(hl.history.format_step(h2.step), "9:99")
        self.assertEqual(hl.history.format_step(h2.step, zero_prefix=True), "0009:000099")

        # Clean up
        shutil.rmtree(OUTPUT_DIR)



if __name__ == "__main__":
    unittest.main()
