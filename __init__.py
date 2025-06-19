from .nodes.nodes import CombineTexts, DescribeImage, TransformText, ClassifyImage

NODE_CLASS_MAPPINGS = {
    'Describe Image': DescribeImage,
    'Combine Texts': CombineTexts,
    'Transform Text': TransformText,
    'Classify Image': ClassifyImage,
}

__all__ = ['NODE_CLASS_MAPPINGS']
