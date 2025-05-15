#%%
import anthropic
import os
## client = anthropic.Anthropic() 
#%%
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

message = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
print(message.content)
# %%
print(message.content[0].text)
