import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import streamlit as st
import pandas as pd

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
selected_option = str(selected_option)

if selected_option == 'January':
  month = 13
elif selected_option == 'February':
  month = 14
elif selected_option == 'March':
  month = 15
elif selected_option == 'April':
  month = 16
elif selected_option == 'May':
  month = 17
elif selected_option == 'June':
  month = 18
elif selected_option == 'July':
  month = 19
elif selected_option == 'August':
  month = 20
elif selected_option == 'September':
  month = 21
elif selected_option == 'October':
  month = 22
elif selected_option == 'November':
  month = 23
elif selected_option == 'December':
  month = 24
else:
  month = 13

user_input = np.array([month], dtype=np.float32)
user_input = Variable(torch.from_numpy(user_input.reshape(-1, 1)))
user_output = new_model(user_input)

st.write("The predicted receipt count for the month of", selected_option, "is", str(user_output.item())[:7], "Million")

months = [i for i in range(13, 25)]
graph_input = np.array(months, dtype=np.float32)
graph_input = Variable(torch.from_numpy(graph_input.reshape(-1, 1)))
graph_output = new_model(graph_input).tolist()
graph_output = sum(graph_output, [])

df = pd.DataFrame({"Months": months, "Receipt_Count_Predicted": graph_output})
st.line_chart(df)