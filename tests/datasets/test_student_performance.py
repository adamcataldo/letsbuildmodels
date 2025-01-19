import unittest
from letsbuildmodels.datasets.student_performance import get_student_data

class TestStudentPerformance(unittest.TestCase):
    def test_get_student_data_default(self):
        train_dataloader, test_dataloader = get_student_data()
        
        # Check if dataloaders are not empty
        self.assertGreater(len(train_dataloader), 0)
        self.assertGreater(len(test_dataloader), 0)
        
        # Check if the batch size is correct
        for batch in train_dataloader:
            self.assertEqual(batch[0].shape[0], 32)
            break  # Only need to check the first batch
        
        for batch in test_dataloader:
            self.assertEqual(batch[0].shape[0], 32)
            break  # Only need to check the first batch

    def test_get_student_data_custom_batch_size(self):
        batch_size = 16
        train_dataloader, test_dataloader = get_student_data(batch_size=batch_size)
        
        # Check if the batch size is correct
        for batch in train_dataloader:
            self.assertEqual(batch[0].shape[0], batch_size)
            break  # Only need to check the first batch
        
        for batch in test_dataloader:
            self.assertEqual(batch[0].shape[0], batch_size)
            break  # Only need to check the first batch

    def test_get_student_data_custom_split(self):
        train_split = 0.7
        train_dataloader, test_dataloader = get_student_data(train_split=train_split)
        
        # Calculate expected sizes
        dataset_size = len(train_dataloader.dataset) + len(test_dataloader.dataset)
        expected_train_size = int(train_split * dataset_size)
        expected_test_size = dataset_size - expected_train_size
        
        # Check if the sizes are correct
        self.assertEqual(len(train_dataloader.dataset), expected_train_size)
        self.assertEqual(len(test_dataloader.dataset), expected_test_size)

if __name__ == '__main__':
    unittest.main()