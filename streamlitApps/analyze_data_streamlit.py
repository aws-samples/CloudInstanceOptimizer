#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: Apache-2.0                                #
######################################################################


import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

# Set the layout configuration
st.set_page_config(layout="wide")

data = pd.read_csv('ec2data.csv')

# Sidebar for user input
st.sidebar.title("Select X and Y Axis")
x_axis = st.sidebar.selectbox("X-Axis", ['Avg_cpu', 'peak_RAM', 'cost', 'runtime','Disk_IO_Read',
                                         'Disk_IO_Write', 'Network_Sent', 'Network_Received'])
y_axis = st.sidebar.selectbox("Y-Axis", ['Avg_cpu', 'peak_RAM', 'cost', 'runtime','Disk_IO_Read',
                                         'Disk_IO_Write', 'Network_Sent', 'Network_Received'])

# Create the scatter plot with hover text
fig = px.scatter(data, x=x_axis, y=y_axis, color='ec2_type', hover_data=['ec2_type', x_axis, y_axis])
fig.update_traces(marker_size=12)
fig.update_layout(width=800, height=600)

# Create the bar chart
bar_chart = data.groupby('ec2_type')[y_axis].agg(['mean', 'sem'])
bar_chart = bar_chart.reset_index()
bar_chart = bar_chart.sort_values('mean')
fig2, ax = plt.subplots(figsize=(6, 4))
ax = bar_chart['mean'].plot(kind='bar', yerr=bar_chart['sem'], capsize=4, ax=ax)
ax.set_xlabel('ec2_type')
ax.set_ylabel(y_axis)
ax.set_xticklabels(bar_chart['ec2_type'], rotation=45, ha='right')  # Label bars with ec2_type

# Display the plots in a single row with two columns
col1, col2 = st.columns(2)
with col1:
    #st.pyplot(fig, use_container_width=True)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.pyplot(fig2, use_container_width=True)