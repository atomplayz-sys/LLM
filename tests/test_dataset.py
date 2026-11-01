from src.data.dataset_loader import format_instruction, get_dataset_info


class TestFormatInstruction:
    def test_basic_format(self, sample_instruction):
        result = format_instruction(sample_instruction)
        assert "What is 2+2?" in result
        assert "4" in result
        assert "<|user|>" in result
        assert "<|assistant|>" in result
        assert "<|system|>" in result

    def test_with_context(self, sample_instruction_with_context):
        result = format_instruction(sample_instruction_with_context)
        assert "Summarize the text" in result
        assert "Context: Some long text here" in result
        assert "Summary" in result

    def test_empty_input_excluded(self, sample_instruction):
        result = format_instruction(sample_instruction)
        assert "Context:" not in result

    def test_whitespace_only_input_excluded(self):
        sample = {"instruction": "Test", "input": "   ", "output": "Result"}
        result = format_instruction(sample)
        assert "Context:" not in result

    def test_template_structure(self, sample_instruction):
        result = format_instruction(sample_instruction)
        assert result.index("<|system|>") < result.index("<|user|>")
        assert result.index("<|user|>") < result.index("<|assistant|>")


class TestDatasetInfo:
    def test_dataset_info_fields(self):
        info = get_dataset_info()
        assert info["name"] == "Open-Platypus"
        assert "garage-bAInd" in info["source"]
        assert "url" in info
        assert "paper" in info

    def test_dataset_info_urls(self):
        info = get_dataset_info()
        assert info["url"].startswith("https://")
        assert info["paper"].startswith("https://")
