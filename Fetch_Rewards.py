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

st.title("Fetch Rewards")
st.header("Estimate Receipt Counts for 2022")

st.write("Select the month of 2022 for which you'd like to see predict the sales")
selected_option = st.selectbox("Choose one", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])

st.write(selected_option, type(str(selected_option)))
month = 13
user_input = np.array([int(month)], dtype=np.float32)
user_input = Variable(torch.from_numpy(user_input.reshape(-1, 1)))
user_output = new_model(user_input)

st.write(user_output)
