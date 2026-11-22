import unittest

from training.train_lora import completion_example


class TemplateTokenizer:
    """Character tokenizer that models the native MiniCPM turn boundary."""
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking=None):
        assert not tokenize
        return "prompt\n" + ("<think>\n\n</think>\n\n" if enable_thinking is False
                             or not add_generation_prompt else "") + (
            "" if add_generation_prompt else messages[-1]["content"] + "<|im_end|>\n")

    def __call__(self, s, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [ord(c) for c in s]}

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


class TestCompletion(unittest.TestCase):
    messages = [{"role": "system", "content": "JSON only"},
                {"role": "user", "content": "mute it"},
                {"role": "assistant", "content": '{"tool":"mute"}'}]

    def test_only_answer_and_turn_end_are_supervised(self):
        tok = TemplateTokenizer()
        ex = completion_example(self.messages, tok, 512)
        self.assertEqual(tok.decode([x for x in ex["labels"] if x != -100]),
                         self.messages[-1]["content"] + "<|im_end|>")
        self.assertEqual(len(ex["labels"]), len(ex["input_ids"]))

    def test_rejects_truncation(self):
        with self.assertRaisesRegex(ValueError, "max"):
            completion_example(self.messages, TemplateTokenizer(), 1)

    def test_rejects_mismatched_token_prefix(self):
        class BrokenTokenizer(TemplateTokenizer):
            def __call__(self, s, *, add_special_tokens):
                ids = super().__call__(s, add_special_tokens=add_special_tokens)["input_ids"]
                if "<|im_end|>" in s:
                    ids[0] = 0
                return {"input_ids": ids}
        with self.assertRaisesRegex(ValueError, "prefix"):
            completion_example(self.messages, BrokenTokenizer(), 512)
