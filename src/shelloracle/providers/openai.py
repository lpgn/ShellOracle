import logging
from collections.abc import AsyncIterator

from openai import APIError, AsyncOpenAI

from shelloracle.providers import Provider, ProviderError, Setting, system_prompt


class OpenAI(Provider):
    name = "OpenAI"

    api_key = Setting(default="")
    base_url = Setting(default="https://api.openai.com/v1")
    model = Setting(default="gpt-4o")

    def __init__(self):
        if not self.api_key:
            msg = "No API key provided"
            raise ProviderError(msg)
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                stream=True,
            )
            async for chunk in stream:
                logging.getLogger(__name__).info(chunk)
                try:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                except:
                    ...
        except APIError as e:
            msg = f"Something went wrong while querying OpenAI: {e}"
            raise ProviderError(msg) from e
