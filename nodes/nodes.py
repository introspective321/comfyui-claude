"""Nodes for the ComfyUI Claude extension."""

from comfyui_types import (
    ChoiceInput,
    ComfyUINode,
    ImageInput,
    StringInput,
    StringOutput,
)

from .ai import describe_image, models, run_prompt

DESCRIBE_IMAGE_PROMPT = 'Describe this image in detail.'
COMBINE_TEXTS_PROMPT = 'Combine the following two texts into one coherent \
prompt without redundancies.'


class DescribeImage(ComfyUINode):
    """Describe an image."""

    category = 'Claude'
    
    image = ImageInput()
    model = ChoiceInput(choices=models)
    api_key = StringInput()

    system_prompt = StringInput(required=False, multiline=True)
    prompt = StringInput(
        required=False, multiline=True, default=DESCRIBE_IMAGE_PROMPT
    )

    description = StringOutput()

    def execute(
        self,
        image: 'torch.Tensor',  # type: ignore[name-defined]  # noqa: F821
        model: str,
        api_key: str,
        system_prompt: str,
        prompt: str,
    ) -> tuple[str, ...]:
        """Send an image to Claude's vision API.

        Args:
            image (torch.Tensor): The image to describe.
            model (str): The model to use.
            api_key (str): The API key to use.
            system_prompt (str): The system prompt to use.
            prompt (str): The prompt to use.

        Returns:
            str: The result of the prompt.
        """
        return (describe_image(image, prompt, system_prompt, model, api_key),)


class CombineTexts(ComfyUINode):
    """Combine two texts."""

    category = 'Claude'

    text_1 = StringInput(multiline=True)
    text_1_prefix = StringInput(default='1')
    text_2 = StringInput(multiline=True)
    text_2_prefix = StringInput(default='2')
    model = ChoiceInput(choices=models)
    api_key = StringInput()

    system_prompt = StringInput(required=False, multiline=True)
    prompt = StringInput(
        required=False, multiline=True, default=DESCRIBE_IMAGE_PROMPT
    )

    combined_texts = StringOutput()

    def execute(
        self,
        text_1: str,
        text_1_prefix: str,
        text_2: str,
        text_2_prefix: str,
        model: str,
        api_key: str,
        system_prompt: str,
        prompt: str,
    ) -> tuple[str, ...]:
        """Combine two texts.

        Args:
            text_1 (str): The first text.
            text_1_prefix (str): The prefix for the first text.
            text_2 (str): The second text.
            text_2_prefix (str): The prefix for the second text.
            model (str): The model to use.
            api_key (str): The API key to use.
            system_prompt (str): The system prompt to use.
            prompt (str): The prompt to use.

        Returns:
            str: The result of the prompt.
        """
        full_prompt = (
            f'{prompt}\n{text_1_prefix} {text_1}\n{text_2_prefix} {text_2}'
        )
        return (run_prompt(full_prompt, system_prompt, model, api_key),)


class TransformText(ComfyUINode):
    """Transform text."""

    category = 'Claude'

    text = StringInput(multiline=True)
    model = ChoiceInput(choices=models)
    api_key = StringInput()

    system_prompt = StringInput(required=False, multiline=True)
    prompt = StringInput(
        required=False, multiline=True, default=DESCRIBE_IMAGE_PROMPT
    )

    transformed_text = StringOutput()

    def execute(
        self,
        text: str,
        model: str,
        api_key: str,
        system_prompt: str,
        prompt: str,
    ) -> tuple[str, ...]:
        """Transform text.

        Args:
            text (str): The text to transform.
            model (str): The model to use.
            api_key (str): The API key to use.
            system_prompt (str): The system prompt to use.
            prompt (str): The prompt to use.

        Returns:
            str: The result of the prompt.
        """
        full_prompt = f'{prompt}\nText: {text}\n'
        return (run_prompt(full_prompt, system_prompt, model, api_key),)


class ClassifyImage(ComfyUINode):
    """Classify an image using Claude."""

    category = 'Claude'
    
    image = ImageInput()
    classification_prompt = StringInput(multiline=True, default="You are a visual classification model. I will provide you with a full-body fashion image containing one or more garments. Your task is to analyze the entire image and classify each distinct visible garment using **only** the following predefined list of classes: dress, jacket, jumpsuit, sweatshirt, coat, cardigan, skort, bikini, leggings, underwear, swimsuit, bodysuit, vest, poncho, saree, kurta, sweater, sleepwear, tank-top, shirt, pants, shorts, thshirt, skirt, topwear, jeans, tshirt, blazer, bra ### Instructions: - Identify the **most prominent garments** visible in the image - Select the **best-fitting label** from the list above for each garment - **MAXIMUM TWO CLASSES ONLY** - return at most 2 garment classifications - If there is only one visible garment, return just the **single class name** - If there are multiple garments, return **only the two most prominent ones** in a **comma-separated format** (e.g., `shirt, pants`) - **Only** use the class names from the list above. Do **not** create new categories or output any additional text - Your output must be **clean and machine-readable**, as it will be passed directly to another model (Moondream) for bounding box extraction ### Output format: - One garment → `tshirt` - Two garments → `tshirt, jeans` - **NEVER return more than two classes** Again, **do not include explanations, descriptions, confidence scores, or formatting**—just the class name(s) as a single plain string with maximum two classes.")
    model = ChoiceInput(choices=models, default='claude-3-7-sonnet-20250219')
    api_key = StringInput()
    system_prompt = StringInput(required=False, multiline=True)

    classification = StringOutput()

    def execute(
        self,
        image: 'torch.Tensor',  # type: ignore[name-defined]  # noqa: F821
        classification_prompt: str,
        model: str,
        api_key: str,
        system_prompt: str,
    ) -> tuple[str, ...]:
        """Classify an image using Claude.

        Args:
            image (torch.Tensor): The image to classify.
            classification_prompt (str): The prompt instructing classification.
            model (str): The model to use.
            api_key (str): The API key to use.
            system_prompt (str): The system prompt to use.

        Returns:
            str: The classification result from Claude.
        """
        return (describe_image(image, classification_prompt, system_prompt, model, api_key),)
