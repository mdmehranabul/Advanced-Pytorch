#%%
from transformers import pipeline
import pandas as pd
# %% classifier
task="zero-shot-classification"
model='facebook/bart-large-mnli'
classifier=pipeline(task=task,model=model)
# %%
documents = ["It was about eleven o’clock in the morning, mid October, with the sun not shining and a look of hard wet rain in the clearness of the foothills. I was wearing my powder-blue suit, with dark blue shirt, tie and display handkerchief, black brogues, black wool socks with dark blue clocks on them. I was neat, clean, shaved and sober, and I didn’t care who knew it. I was everything the well-dressed private detective ought to be. I was calling on four million dollars.",
             "When Mr. Bilbo Baggins of Bag End announced that he would shortly be celebrating his eleventy-first birthday with a party of special magnificence, there was much talk and excitement in Hobbiton.",
             "Welcome. And congratulations. I am delighted that you could make it. Getting here wasn’t easy, I know. In fact, I suspect it was a little tougher than you realize. To begin with, for you to be here now trillions of drifting atoms had somehow to assemble in an intricate and curiously obliging manner to create you. It’s an arrangement so specialized and particular that it has never been tried before and will only exist this once."
             ]
# %%
candidate_labels=["crime","fantasy","history"]
# %%
res=classifier(documents,candidate_labels=candidate_labels)
# %%
res
# %%
res[1]
# %%
pd.DataFrame(res[1]).plot.bar(x='labels',y='scores',rot=0,title='The Lord of the Rings')
# %% Flag multiple labels
classifier(documents,candidate_labels=candidate_labels, multi_class=True)

# %%
