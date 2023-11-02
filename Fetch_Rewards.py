import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import streamlit as st

class regressionModel(nn.Module):
  def __init__(self):
    super(regressionModel, self).__init__()
    self.fc1 = nn.Linear(1, 1)

  def forward(self, x):
    out = self.fc1(x)
    return out

new_model = regressionModel()
new_model.load_state_dict(torch.load('fetch_model.pkl'))

month = st.text_input("Enter month of 2022 for which we want to predict the sales: ")

user_input = np.array([month], dtype=np.float32)
user_input = Variable(torch.from_numpy(user_input.reshape(-1, 1)))
user_output = new_model(user_input)

st.title("Fetch Rewards")
st.header("Estimate Receipt Counts for 2022")

search_query = st.text_input("Search Offers by Category, Brand or Retailer")

st.write(user_output)
