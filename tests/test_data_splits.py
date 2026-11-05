import unittest
import pandas as pd
from data_splits import clean_emotion_data, split_emotion_data

class DatasetSplitTests(unittest.TestCase):
    def test_duplicate_and_conflicting_labels_removed(self):
        data = pd.DataFrame({"text": [" Hello ", "hello", "hello", "bye", " bye "],
                             "label": [0, 0, 1, 1, 1]})
        cleaned, info = clean_emotion_data(data)
        self.assertEqual(cleaned["text"].tolist(), ["bye"])
        self.assertEqual(info["removed_conflicting_rows"], 3)
        self.assertEqual(info["removed_duplicate_rows"], 1)

    def test_no_train_validation_test_overlap(self):
        data = pd.DataFrame({"text": [f"emotion example {i}" for i in range(120)],
                             "label": [i % 6 for i in range(120)]})
        (train, val, test, yt, yv, ys), stats = split_emotion_data(data)
        self.assertEqual(stats["remaining_rows"], 120)
        self.assertFalse(set(train) & set(val))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(val) & set(test))
        self.assertEqual([len(train), len(val), len(test)], [96, 12, 12])
        self.assertEqual([len(yt), len(yv), len(ys)], [96, 12, 12])

if __name__ == "__main__":
    unittest.main()
