"""
AutoStent Web Application
A modern web UI for RL-based stent design optimization.
Uses Streamlit for rapid development with beautiful UI.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
import sys
from pathlib import Path

# Add autostent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "autostent"))

from autostent.geometry import SplineStentGeometry, StentGeometryParameters
from autostent.rl import StentDesignEnv
from autostent.simulation import FourCSimulator

# Page config
st.set_page_config(
    page_title="AutoStent - RL Design Optimization",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">AutoStent Design Optimization</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Reinforcement Learning for Autonomous Stent Design</p>', unsafe_allow_html=True)

# Initialize session state
if 'env' not in st.session_state:
    st.session_state.env = StentDesignEnv(fourc_executable=None, max_episode_steps=50)
    st.session_state.obs, st.session_state.info = st.session_state.env.reset()
    st.session_state.episode_rewards = []
    st.session_state.step_count = 0
    st.session_state.training_history = []

# Sidebar - Design Parameters
st.sidebar.header("🔧 Design Parameters")

with st.sidebar.expander("Stent Geometry", expanded=True):
    diameter = st.slider("Diameter (mm)", 6.0, 14.0, 
                        float(st.session_state.info['parameters']['diameter']), 0.5)
    length = st.slider("Length (mm)", 10.0, 30.0,
                      float(st.session_state.info['parameters']['length']), 1.0)
    strut_thickness = st.slider("Strut Thickness (mm)", 0.05, 0.20,
                               float(st.session_state.info['parameters']['strut_thickness']), 0.01)
    strut_width = st.slider("Strut Width (mm)", 0.10, 0.30,
                           float(st.session_state.info['parameters']['strut_width']), 0.01)
    num_struts = st.slider("Number of Struts", 6, 18,
                          int(st.session_state.info['parameters']['num_struts']), 1)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Stent Visualization")
    
    # 3D Stent visualization
    params = st.session_state.info['parameters']
    
    # Create 3D stent mesh
    theta = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, params['length'], 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    r = params['diameter'] / 2
    
    x = r * np.cos(theta_grid)
    y = r * np.sin(theta_grid)
    
    fig_3d = go.Figure()
    
    # Stent surface
    fig_3d.add_trace(go.Surface(
        x=x, y=y, z=z_grid,
        colorscale='Blues',
        opacity=0.7,
        showscale=False,
        name='Stent Surface'
    ))
    
    # Struts
    for i in range(int(params['num_struts'])):
        angle = 2 * np.pi * i / params['num_struts']
        strut_x = (r + 0.2) * np.cos(angle) * np.ones(50)
        strut_y = (r + 0.2) * np.sin(angle) * np.ones(50)
        strut_z = np.linspace(0, params['length'], 50)
        
        fig_3d.add_trace(go.Scatter3d(
            x=strut_x, y=strut_y, z=strut_z,
            mode='lines',
            line=dict(color='red', width=8),
            name=f'Strut {i+1}' if i == 0 else None,
            showlegend=(i == 0)
        ))
    
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"Stent: D={params['diameter']:.1f}mm, L={params['length']:.1f}mm, {int(params['num_struts'])} struts"
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)

with col2:
    st.subheader("📈 Performance Metrics")
    
    # Get current simulation result
    result = st.session_state.info.get('result', {})
    stress = result.get('max_von_mises_stress', 150.0)
    displacement = result.get('max_displacement', 0.5)
    
    # Metrics
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Max Stress", f"{stress:.1f} MPa", 
                 delta=f"{stress - 150:.1f}" if stress != 150 else None,
                 delta_color="inverse")
    with col_m2:
        st.metric("Max Displacement", f"{displacement:.2f} mm",
                 delta=f"{displacement - 0.5:.2f}" if displacement != 0.5 else None,
                 delta_color="inverse")
    
    # Safety indicator
    safety_score = max(0, min(100, 100 - (stress - 100) / 2))
    st.progress(int(safety_score) / 100)
    st.caption(f"Safety Score: {safety_score:.0f}%")
    
    # Optimal targets
    st.markdown("---")
    st.markdown("**🎯 Optimal Targets:**")
    st.markdown("""
    - Stress: < 150 MPa
    - Displacement: < 0.5 mm
    - Strut thickness: ~0.12 mm
    - Diameter: ~10 mm
    """)

# RL Training Section
st.markdown("---")
st.subheader("🤖 RL Training")

col_train1, col_train2, col_train3 = st.columns([1, 1, 2])

with col_train1:
    if st.button("🎲 Random Step", use_container_width=True):
        action = st.session_state.env.action_space.sample()
        obs, reward, done, trunc, info = st.session_state.env.step(action)
        st.session_state.obs = obs
        st.session_state.info = info
        st.session_state.step_count += 1
        st.session_state.episode_rewards.append(reward)
        if done or trunc:
            st.session_state.training_history.append(sum(st.session_state.episode_rewards))
            st.session_state.episode_rewards = []
            st.session_state.obs, st.session_state.info = st.session_state.env.reset()
        st.rerun()

with col_train2:
    if st.button("🔄 Reset Environment", use_container_width=True):
        st.session_state.obs, st.session_state.info = st.session_state.env.reset()
        st.session_state.episode_rewards = []
        st.session_state.step_count = 0
        st.rerun()

with col_train3:
    num_steps = st.slider("Training steps", 100, 5000, 1000, 100)
    if st.button("🚀 Train RL Agent", use_container_width=True):
        progress_bar = st.progress(0)
        status = st.empty()
        
        for i in range(num_steps):
            action = st.session_state.env.action_space.sample()  # Would use trained model
            obs, reward, done, trunc, info = st.session_state.env.step(action)
            st.session_state.episode_rewards.append(reward)
            
            if done or trunc:
                st.session_state.training_history.append(sum(st.session_state.episode_rewards))
                st.session_state.episode_rewards = []
                obs, info = st.session_state.env.reset()
            
            if i % 50 == 0:
                progress_bar.progress(i / num_steps)
                status.text(f"Step {i}/{num_steps} | Episodes: {len(st.session_state.training_history)}")
        
        st.session_state.obs = obs
        st.session_state.info = info
        progress_bar.progress(1.0)
        status.text("✓ Training complete!")
        st.rerun()

# Training History Plot
if st.session_state.training_history:
    st.subheader("📉 Training Progress")
    
    fig_history = make_subplots(rows=1, cols=2, 
                                subplot_titles=('Episode Rewards', 'Moving Average'))
    
    # Raw rewards
    fig_history.add_trace(
        go.Scatter(y=st.session_state.training_history, mode='lines',
                  line=dict(color='lightblue'), name='Episode Reward'),
        row=1, col=1
    )
    
    # Moving average
    if len(st.session_state.training_history) > 10:
        window = 10
        moving_avg = np.convolve(st.session_state.training_history, 
                                 np.ones(window)/window, mode='valid')
        fig_history.add_trace(
            go.Scatter(y=moving_avg, mode='lines',
                      line=dict(color='blue', width=2), name='Moving Avg (10)'),
            row=1, col=2
        )
    
    fig_history.update_layout(height=300, showlegend=True)
    st.plotly_chart(fig_history, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9rem;'>
    AutoStent - Research Tool for Autonomous Stent Design | 
    Built with Streamlit & AutoStent Framework
</div>
""", unsafe_allow_html=True)

