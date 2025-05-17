#%%
import openai
import os
# %%
openai.api_key = os.getenv("OPENAI_API_KEY")
# %%
response=openai.images.generate(prompt="Black dog with a tiny hat",n=1,size="1024x1024")
image_url=response.data[0].url
image_url
# %%
