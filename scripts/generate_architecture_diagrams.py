#!/usr/bin/env python3
"""
Generate architecture diagrams for the comprehensive report
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np

def create_system_architecture_diagram():
    """Create the main system architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define colors
    frontend_color = '#3498db'
    api_color = '#9b59b6'
    core_color = '#e74c3c'
    ml_color = '#f39c12'
    data_color = '#27ae60'
    
    # Frontend Layer
    frontend_y = 8
    ax.add_patch(Rectangle((1, frontend_y), 3, 1.5, facecolor=frontend_color, edgecolor='black', alpha=0.7))
    ax.text(2.5, frontend_y+0.75, 'Next.js\nFrontend', ha='center', va='center', fontsize=10, weight='bold')
    
    ax.add_patch(Rectangle((5, frontend_y), 3, 1.5, facecolor=frontend_color, edgecolor='black', alpha=0.7))
    ax.text(6.5, frontend_y+0.75, 'Streamlit\nDashboard', ha='center', va='center', fontsize=10, weight='bold')
    
    ax.add_patch(Rectangle((9, frontend_y), 3, 1.5, facecolor=frontend_color, edgecolor='black', alpha=0.7))
    ax.text(10.5, frontend_y+0.75, 'Monitoring\n(Grafana)', ha='center', va='center', fontsize=10, weight='bold')
    
    # API Gateway
    api_y = 6
    ax.add_patch(Rectangle((3, api_y), 7, 1.5, facecolor=api_color, edgecolor='black', alpha=0.7))
    ax.text(6.5, api_y+0.75, 'API Gateway (FastAPI + Auth)', ha='center', va='center', fontsize=11, weight='bold')
    
    # Core Trading Engine
    core_y = 3.5
    ax.add_patch(Rectangle((1, core_y), 11, 2, facecolor=core_color, edgecolor='black', alpha=0.7))
    ax.text(6.5, core_y+1.7, 'Core Trading Engine', ha='center', va='center', fontsize=12, weight='bold')
    
    # Core components
    ax.add_patch(Rectangle((1.5, core_y+0.2), 3, 1.2, facecolor='white', edgecolor='black', alpha=0.9))
    ax.text(3, core_y+0.8, 'Strategy\nManager', ha='center', va='center', fontsize=9)
    
    ax.add_patch(Rectangle((5, core_y+0.2), 3, 1.2, facecolor='white', edgecolor='black', alpha=0.9))
    ax.text(6.5, core_y+0.8, 'Risk\nManager', ha='center', va='center', fontsize=9)
    
    ax.add_patch(Rectangle((8.5, core_y+0.2), 3, 1.2, facecolor='white', edgecolor='black', alpha=0.9))
    ax.text(10, core_y+0.8, 'Execution\nEngine', ha='center', va='center', fontsize=9)
    
    # ML/AI Layer
    ml_y = 1
    ax.add_patch(Rectangle((1, ml_y), 11, 1.5, facecolor=ml_color, edgecolor='black', alpha=0.7))
    ax.text(6.5, ml_y+1.2, 'ML/AI Layer', ha='center', va='center', fontsize=11, weight='bold')
    
    # ML components
    ax.text(3, ml_y+0.5, 'LSTM', ha='center', va='center', fontsize=9)
    ax.text(6.5, ml_y+0.5, 'XGBoost', ha='center', va='center', fontsize=9)
    ax.text(10, ml_y+0.5, 'RL Agents', ha='center', va='center', fontsize=9)
    
    # Data Pipeline
    data_y = -0.5
    ax.add_patch(Rectangle((1, data_y), 11, 1, facecolor=data_color, edgecolor='black', alpha=0.7))
    ax.text(6.5, data_y+0.5, 'Data Pipeline', ha='center', va='center', fontsize=11, weight='bold')
    
    # Draw connections
    # Frontend to API
    ax.arrow(2.5, frontend_y, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
    ax.arrow(6.5, frontend_y, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
    ax.arrow(10.5, frontend_y, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # API to Core
    ax.arrow(6.5, api_y, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Core to ML
    ax.arrow(3, core_y, 0, -0.8, head_width=0.2, head_length=0.1, fc='black', ec='black')
    ax.arrow(6.5, core_y, 0, -0.8, head_width=0.2, head_length=0.1, fc='black', ec='black')
    ax.arrow(10, core_y, 0, -0.8, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # ML to Data
    ax.arrow(6.5, ml_y, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # Configure plot
    ax.set_xlim(0, 13)
    ax.set_ylim(-1, 10)
    ax.axis('off')
    ax.set_title('AsterAI System Architecture', fontsize=16, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_data_flow_diagram():
    """Create data flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Data sources
    sources_x = 1
    ax.add_patch(Rectangle((sources_x, 6), 2.5, 1, facecolor='#3498db', edgecolor='black', alpha=0.7))
    ax.text(sources_x+1.25, 6.5, 'Aster DEX', ha='center', va='center', fontsize=10, weight='bold')
    
    ax.add_patch(Rectangle((sources_x, 4.5), 2.5, 1, facecolor='#3498db', edgecolor='black', alpha=0.7))
    ax.text(sources_x+1.25, 5, 'Binance', ha='center', va='center', fontsize=10, weight='bold')
    
    ax.add_patch(Rectangle((sources_x, 3), 2.5, 1, facecolor='#3498db', edgecolor='black', alpha=0.7))
    ax.text(sources_x+1.25, 3.5, 'Historical', ha='center', va='center', fontsize=10, weight='bold')
    
    # Processing pipeline
    process_x = 5
    ax.add_patch(Rectangle((process_x, 5.5), 3, 1.5, facecolor='#9b59b6', edgecolor='black', alpha=0.7))
    ax.text(process_x+1.5, 6.25, 'Data Router\n& Validator', ha='center', va='center', fontsize=10, weight='bold')
    
    ax.add_patch(Rectangle((process_x, 3.5), 3, 1.5, facecolor='#9b59b6', edgecolor='black', alpha=0.7))
    ax.text(process_x+1.5, 4.25, 'Normalization\n& Cleaning', ha='center', va='center', fontsize=10, weight='bold')
    
    ax.add_patch(Rectangle((process_x, 1.5), 3, 1.5, facecolor='#9b59b6', edgecolor='black', alpha=0.7))
    ax.text(process_x+1.5, 2.25, 'Feature\nEngineering', ha='center', va='center', fontsize=10, weight='bold')
    
    # Output
    output_x = 9.5
    ax.add_patch(Rectangle((output_x, 3.5), 2.5, 3, facecolor='#e74c3c', edgecolor='black', alpha=0.7))
    ax.text(output_x+1.25, 5, 'ML Models\n&\nTrading\nEngine', ha='center', va='center', fontsize=10, weight='bold')
    
    # Arrows
    # Sources to processing
    ax.arrow(sources_x+2.5, 6.5, 1, -0.25, head_width=0.15, head_length=0.2, fc='black', ec='black')
    ax.arrow(sources_x+2.5, 5, 1, 0.5, head_width=0.15, head_length=0.2, fc='black', ec='black')
    ax.arrow(sources_x+2.5, 3.5, 1, 0.5, head_width=0.15, head_length=0.2, fc='black', ec='black')
    
    # Processing stages
    ax.arrow(process_x+1.5, 5.5, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
    ax.arrow(process_x+1.5, 3.5, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # To output
    ax.arrow(process_x+3, 2.25, 1, 2.5, head_width=0.15, head_length=0.2, fc='black', ec='black')
    
    # Labels
    ax.text(2.25, 7.5, 'External Data Sources', fontsize=12, weight='bold')
    ax.text(5.5, 7.5, 'Processing Pipeline', fontsize=12, weight='bold')
    ax.text(9.75, 7.5, 'Trading System', fontsize=12, weight='bold')
    
    # Configure plot
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Data Flow Architecture', fontsize=16, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('data_flow_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_trading_strategy_diagram():
    """Create trading strategy comparison diagram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Strategy performance comparison
    strategies = ['Grid\nTrading', 'Volatility\nTrading', 'Hybrid\nAdaptive', 'Buy &\nHold']
    returns = [12.3, 18.7, 15.8, 7.2]
    sharpe = [1.87, 2.24, 2.41, 0.93]
    colors = ['#3498db', '#e74c3c', '#f39c12', '#95a5a6']
    
    x = np.arange(len(strategies))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, returns, width, label='Return (%)', color=colors, alpha=0.8)
    bars2 = ax1.bar(x + width/2, [s*5 for s in sharpe], width, label='Sharpe Ratio (x5)', color=colors, alpha=0.6)
    
    ax1.set_xlabel('Strategy', fontsize=12, weight='bold')
    ax1.set_ylabel('Performance Metrics', fontsize=12, weight='bold')
    ax1.set_title('Strategy Performance Comparison', fontsize=14, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Risk-Return scatter
    ax2.scatter(sharpe, returns, s=200, c=colors, alpha=0.8, edgecolors='black')
    
    for i, txt in enumerate(strategies):
        ax2.annotate(txt.replace('\n', ' '), (sharpe[i], returns[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax2.set_xlabel('Sharpe Ratio', fontsize=12, weight='bold')
    ax2.set_ylabel('Return (%)', fontsize=12, weight='bold')
    ax2.set_title('Risk-Adjusted Returns', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add quadrant lines
    ax2.axhline(y=np.mean(returns), color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=np.mean(sharpe), color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('trading_strategies_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_ml_pipeline_diagram():
    """Create ML pipeline diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Pipeline stages
    stages = [
        ('Raw Data', 1, '#3498db'),
        ('Feature\nEngineering', 3, '#9b59b6'),
        ('Model\nTraining', 5, '#e74c3c'),
        ('Validation', 7, '#f39c12'),
        ('Deployment', 9, '#27ae60')
    ]
    
    y_center = 3
    box_height = 1.5
    box_width = 1.8
    
    for i, (label, x, color) in enumerate(stages):
        # Main boxes
        ax.add_patch(Rectangle((x-box_width/2, y_center-box_height/2), box_width, box_height, 
                              facecolor=color, edgecolor='black', alpha=0.7))
        ax.text(x, y_center, label, ha='center', va='center', fontsize=11, weight='bold', color='white')
        
        # Sub-components
        if i < len(stages) - 1:
            ax.arrow(x+box_width/2, y_center, 1.2-box_width, 0, 
                    head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    # Add sub-labels
    sub_labels = [
        (1, 1.5, 'Quality\nChecks'),
        (3, 1.5, '41+\nIndicators'),
        (5, 1.5, 'GPU\nAccelerated'),
        (7, 1.5, 'Backtest\nTesting'),
        (9, 1.5, 'Live\nTrading')
    ]
    
    for x, y, label in sub_labels:
        ax.text(x, y, label, ha='center', va='center', fontsize=9, style='italic')
    
    # Model types
    model_y = 4.5
    models = [
        (4, 'LSTM'),
        (5, 'XGBoost'),
        (6, 'Ensemble')
    ]
    
    for x, model in models:
        ax.add_patch(Rectangle((x-0.5, model_y), 1, 0.6, 
                              facecolor='white', edgecolor='black', alpha=0.9))
        ax.text(x, model_y+0.3, model, ha='center', va='center', fontsize=9)
    
    # Configure plot
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('ML Model Training Pipeline', fontsize=16, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('ml_pipeline_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_risk_management_diagram():
    """Create risk management framework diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Risk levels
    levels = [
        ('Portfolio Level', 2, 6, '#e74c3c', [
            'Max Drawdown < 20%',
            'Daily VaR < 5%',
            'Leverage < 3x'
        ]),
        ('Position Level', 5.5, 6, '#f39c12', [
            'Position Size < 5% NAV',
            'Correlation Limits',
            'Concentration < 30%'
        ]),
        ('Trade Level', 9, 6, '#3498db', [
            'Stop Loss 1-3%',
            'Take Profit 2-10%',
            'Time Limits 24-48h'
        ])
    ]
    
    for title, x, y, color, rules in levels:
        # Main box
        ax.add_patch(Rectangle((x-1.5, y-2), 3, 3, facecolor=color, edgecolor='black', alpha=0.3))
        ax.text(x, y+0.8, title, ha='center', va='center', fontsize=12, weight='bold')
        
        # Rules
        for i, rule in enumerate(rules):
            ax.text(x, y-0.5-i*0.5, rule, ha='center', va='center', fontsize=9)
    
    # Risk flow arrows
    ax.arrow(3.5, 5, 1, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax.arrow(7, 5, 1, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    # Central monitoring
    ax.add_patch(Circle((5.5, 2), 1.5, facecolor='#27ae60', edgecolor='black', alpha=0.5))
    ax.text(5.5, 2, 'Real-time\nRisk\nMonitoring', ha='center', va='center', fontsize=11, weight='bold')
    
    # Connections to monitoring
    for x in [2, 5.5, 9]:
        ax.plot([x, 5.5], [4, 3.5], 'k--', alpha=0.5)
    
    # Configure plot
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Risk Management Framework', fontsize=16, weight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('risk_management_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    """Generate all architecture diagrams."""
    print("📊 Generating architecture diagrams...")
    
    create_system_architecture_diagram()
    print("✅ Created system architecture diagram")
    
    create_data_flow_diagram()
    print("✅ Created data flow diagram")
    
    create_trading_strategy_diagram()
    print("✅ Created trading strategy diagram")
    
    create_ml_pipeline_diagram()
    print("✅ Created ML pipeline diagram")
    
    create_risk_management_diagram()
    print("✅ Created risk management diagram")
    
    print("\n✨ All diagrams generated successfully!")
    print("📁 Diagrams saved in current directory")

if __name__ == "__main__":
    main()
