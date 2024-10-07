#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: Apache-2.0                                #
######################################################################


import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
fig1 = px.scatter(data, x=x_axis, y=y_axis, color='ec2_type', hover_data=['ec2_type', x_axis, y_axis])
fig1.update_traces(marker_size=12)
fig1.update_layout(width=800, height=600)

# Create the bar chart data
bar_chart = data.groupby('ec2_type')[y_axis].agg(['mean', 'sem'])
bar_chart = bar_chart.reset_index()
bar_chart = bar_chart.sort_values('mean')

# Create a horizontal bar chart with error bars using Plotly
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    y=bar_chart['ec2_type'],
    x=bar_chart['mean'],
    orientation='h',
    error_x=dict(type='data', array=bar_chart['sem'], visible=True),
    marker_color='blue'
))

# Set the number of items to display at once
items_per_page = 20

# Calculate the number of pages
num_pages = -(-len(bar_chart) // items_per_page)  # Ceiling division

# Create buttons for scrolling
buttons = []
for i in range(num_pages):
    buttons.append(
        dict(
            args=[{"yaxis.range": [len(bar_chart) - (i+1)*items_per_page, len(bar_chart) - i*items_per_page]}],
            label=f"Page {i+1}",
            method="relayout"
        )
    )

fig2.update_layout(
    title=f'{y_axis} by EC2 Type',
    xaxis_title=y_axis,
    yaxis_title='EC2 Type',
    height=600,
    width=800,
    yaxis=dict(
        autorange="reversed",
        tickfont=dict(size=10),
        range=[len(bar_chart) - items_per_page, len(bar_chart)]
    ),
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.05,
            showactive=True,
            buttons=buttons
        )
    ]
)

# Display the plots in a single row with two columns
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)