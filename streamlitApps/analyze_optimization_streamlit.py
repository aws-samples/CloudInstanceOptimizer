#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: Apache-2.0                                #
######################################################################

import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import pickle

# Set the layout configuration
st.set_page_config(layout="wide")

if st.button('Reload Data'):
    st.experimental_rerun()

with open("optimizer_state.pkl", 'rb') as f:
    opt = pickle.load(f)

if isinstance(opt, dict):

    Xi = np.array(opt['Xi'])
    Xi = Xi.reshape(Xi.shape[0]*Xi.shape[1], Xi.shape[-1])
    arr = np.array(opt['yi'])
    arr = arr.reshape(arr.shape[0]*arr.shape[1])
    argmin = np.argmin(arr)
else:
    argmin = np.argmin(opt.yi)
    Xi = np.array(opt.Xi)
    arr = np.array(opt.yi)
ncols = Xi.shape[1]
nsamples = Xi.shape[0]

data = Xi.copy()
data = np.hstack((data,arr.reshape((-1,1))  ))

colvars = ['var_' + str(i) for i in range(ncols)]
colvars.extend(["Optimization Metric", "Sequential Sample"])


data = np.hstack((data,np.arange(1, nsamples+1).reshape((-1,1))  ))

data = pd.DataFrame(data)
data.columns = colvars

# Reorder the DataFrame
all_columns = data.columns.tolist()
all_columns.remove('Optimization Metric')
all_columns.remove('Sequential Sample')
new_column_order = ['Optimization Metric', 'Sequential Sample'] + all_columns
data = data[new_column_order]
colvars = data.columns

# Sidebar for user input
st.sidebar.title("Select X and Y Axis")
x_axis = st.sidebar.selectbox("X-Axis", colvars)
y_axis = st.sidebar.selectbox("Y-Axis", colvars)

# Add slider for selecting top N rows
n_rows = st.sidebar.slider("Select number of top rows to display", 1, 100, 10)

st.sidebar.subheader("Y-Axis Limits")
y_min = st.sidebar.number_input("Y-Axis Minimum", value=float(data[y_axis].min()))
y_max = st.sidebar.number_input("Y-Axis Maximum", value=float(data[y_axis].max()))

# Create a color array for the scatter plot
colors = ['blue'] * len(data)
min_index = data['Optimization Metric'].idxmin()
colors[min_index] = 'red'

# Create the scatter plot with hover text and custom colors
fig = px.scatter(data, x=x_axis, y=y_axis, color=colors, color_discrete_map={'blue': 'blue', 'red': 'red'})
fig.update_traces(marker_size=12)
fig.update_layout(width=800, height=600)

# Set y-axis limits
fig.update_yaxes(range=[y_min, y_max])

# Remove the color legend
fig.update_layout(showlegend=False)

# Display the plots in a single row with two columns
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig, use_container_width=True)

    # Add notation about the red point
    st.markdown("**Note:** The red point represents the data point with the smallest Optimization Metric.")

# Display the minimum Optimization Metric value
min_metric = data['Optimization Metric'].min()
st.sidebar.markdown(f"**Minimum Optimization Metric:** {min_metric:.4f}")
st.sidebar.markdown(f"**Total Job Executions:** {nsamples:.0f}")

# Display the best X values
best_x_values = data.loc[min_index, [f'var_{i}' for i in range(ncols)]]
st.sidebar.markdown("**Best X Values:**")
for var, value in best_x_values.items():
    st.sidebar.markdown(f"- {var}: {value:.4f}")


# Sort the dataframe by Optimization Metric and display top N rows
sorted_data = data.sort_values('Optimization Metric')
st.subheader(f"Top {n_rows} Rows Sorted by Optimization Metric")
st.dataframe(sorted_data.head(n_rows), use_container_width=True)