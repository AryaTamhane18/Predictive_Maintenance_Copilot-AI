import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# LLaVA — vision model
llm_vision = ChatOllama(model="llava")


def analyse_machine_image(image_path: str, context: str = "") -> str:
    """
    Use LLaVA to visually inspect a machine image.
    Returns a plain-English description of visible damage or wear.
    """
    # Read and encode image as base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    prompt = f"""You are a bearing and industrial equipment inspector.
Look at this machine image carefully and identify:
1. Any visible damage, wear, or corrosion
2. Unusual discolouration or surface defects
3. Misalignment or structural issues
4. Overall condition (Good / Fair / Poor / Critical)

{f'Additional context: {context}' if context else ''}

Be specific and technical in your assessment."""

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"}
        ]
    )

    response = llm_vision.invoke([message])
    return response.content.strip()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: PYTHONPATH=. python tools/vision.py <image_path>")
    else:
        result = analyse_machine_image(path)
        print(f"\nVisual Inspection Result:\n{result}")