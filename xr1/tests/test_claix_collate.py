import unittest
from unittest.mock import patch

import torch

from mibot.data.collate.custom_collate import CustomCollate, IMAGE_ID, IM_START_ID, STATE_ID


class FakeProcessor:
    class Tokenizer:
        def convert_tokens_to_ids(self, values):
            return [151669,151670,151671,151730]
    tokenizer = Tokenizer()
    class ImageProcessor:
        merge_size = 1
    image_processor = ImageProcessor()
    def apply_chat_template(self, messages, **kwargs):
        length, views = messages
        tokens = [IM_START_ID] + [IMAGE_ID]*views + [IM_START_ID,STATE_ID]
        return dict(input_ids=torch.tensor([tokens+[0]*(length-len(tokens))]),
                    pixel_values=torch.ones(views,3),image_grid_thw=torch.ones(views,3,dtype=torch.long))


def sample(length=413,views=2):
    return dict(messages=(length,views),action=torch.zeros(30,60),action_mask=torch.ones(30,60),
                state=torch.zeros(1,60),vlm_action_target=torch.zeros(30,60),
                vlm_action_mask=torch.ones(30,60),vlm_action_actual_length=30)


class PackingTest(unittest.TestCase):
    def make_collator(self):
        with patch('mibot.data.collate.custom_collate.AutoProcessor.from_pretrained',return_value=FakeProcessor()):
            return CustomCollate(strict_samples=True,expected_views=2)

    def test_certified_worst_case_pack(self):
        collator=self.make_collator()
        batch=collator([sample() for _ in range(24)])
        self.assertEqual(batch['input_ids'].shape,(1,9912))
        self.assertEqual(batch['action_segments'].shape,(24,2))
        self.assertEqual(batch['image_grid_thw'].shape,(48,3))
        self.assertEqual(batch['action'].shape,(24,30,60))

    def test_overflow_and_view_and_certification_fail_closed(self):
        collator=self.make_collator();collator.max_length=4096
        with self.assertRaisesRegex(ValueError,'drop sample'):
            collator([sample() for _ in range(24)])
        collator.max_length=12288
        with self.assertRaisesRegex(ValueError,'413-token'):
            collator([sample(414)])
        with self.assertRaisesRegex(ValueError,'view count'):
            collator([sample(413,1)])
        with self.assertRaisesRegex(ValueError,'no training sample'):
            collator([])
