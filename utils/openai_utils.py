def call_openai(prompt: str, system_message: str, model: str, temperature: float, client) -> str:
    """
    A unified wrapper for making OpenAI API calls across all modules.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    
    return response.choices[0].message.content