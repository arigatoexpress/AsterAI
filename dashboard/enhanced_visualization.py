"""
Enhanced Data Visualization Module
Rich, real-time charts for trading system monitoring.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from dash import html, dcc
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class EnhancedVisualizer:
    """Advanced visualization for HFT system monitoring."""

    def __init__(self):
        self.colors = {
            'profit': '#00ff00',
            'loss': '#ff0000',
            'neutral': '#ffff00',
            'background': '#1e1e1e',
            'text': '#ffffff'
        }

    def create_pnl_chart(self, pnl_data: pd.DataFrame) -> go.Figure:
        """Create interactive PnL chart with drawdown overlay."""
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # PnL line
        fig.add_trace(
            go.Scatter(
                x=pnl_data.index,
                y=pnl_data['pnl'],
                name='PnL',
                line=dict(color=self.colors['profit'], width=2)
            ),
            secondary_y=False
        )

        # Cumulative PnL
        fig.add_trace(
            go.Scatter(
                x=pnl_data.index,
                y=pnl_data['pnl'].cumsum(),
                name='Cumulative PnL',
                line=dict(color=self.colors['neutral'], width=1)
            ),
            secondary_y=True
        )

        # Drawdown area
        drawdown = pnl_data['pnl'].cumsum() - pnl_data['pnl'].cumsum().expanding().max()
        fig.add_trace(
            go.Scatter(
                x=pnl_data.index,
                y=drawdown,
                name='Drawdown',
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.3)',
                line=dict(color=self.colors['loss'])
            ),
            secondary_y=False
        )

        fig.update_layout(
            title='Real-time PnL & Drawdown',
            template='plotly_dark',
            hovermode='x unified'
        )

        return fig

    def create_vpin_heatmap(self, vpin_data: pd.DataFrame) -> go.Figure:
        """VPIN toxicity heatmap across assets."""
        fig = go.Figure(data=go.Heatmap(
            z=vpin_data.values,
            x=vpin_data.columns,
            y=vpin_data.index.strftime('%H:%M'),
            colorscale='RdYlGn_r',  # Red for toxic, green for clean
            hoverongaps=False
        ))

        fig.update_layout(
            title='VPIN Toxicity Heatmap',
            template='plotly_dark'
        )

        return fig

    def create_risk_metrics_dashboard(self, metrics: Dict) -> html.Div:
        """Create risk metrics dashboard with gauges."""
        return html.Div([
            html.H3("Risk Metrics Dashboard", style={'color': self.colors['text']}),
            html.Div([
                dcc.Graph(
                    figure=self._create_gauge_chart(
                        'Sharpe Ratio', metrics.get('sharpe', 0), 0, 5, 2.0
                    ),
                    style={'width': '48%', 'display': 'inline-block'}
                ),
                dcc.Graph(
                    figure=self._create_gauge_chart(
                        'Win Rate %', metrics.get('win_rate', 0)*100, 0, 100, 55
                    ),
                    style={'width': '48%', 'display': 'inline-block'}
                )
            ]),
            html.Div([
                dcc.Graph(
                    figure=self._create_gauge_chart(
                        'Drawdown %', abs(metrics.get('drawdown', 0))*100, 0, 50, 15
                    ),
                    style={'width': '48%', 'display': 'inline-block'}
                ),
                dcc.Graph(
                    figure=self._create_gauge_chart(
                        'Profit Factor', metrics.get('profit_factor', 1), 0.5, 3, 1.8
                    ),
                    style={'width': '48%', 'display': 'inline-block'}
                )
            ])
        ])

    def _create_gauge_chart(self, title: str, value: float, min_val: float, max_val: float, target: float) -> go.Figure:
        """Create gauge chart for metrics."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': target
                }
            }
        ))

        fig.update_layout(template='plotly_dark')
        return fig

    def create_order_book_visualization(self, order_book: Dict) -> go.Figure:
        """Visualize order book depth."""
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        fig = go.Figure()

        if bids:
            fig.add_trace(go.Bar(
                x=[p for p, v in bids],
                y=[v for p, v in bids],
                name='Bids',
                marker_color=self.colors['profit'],
                orientation='h'
            ))

        if asks:
            fig.add_trace(go.Bar(
                x=[p for p, v in asks],
                y=[v for p, v in asks],
                name='Asks',
                marker_color=self.colors['loss'],
                orientation='h'
            ))

        fig.update_layout(
            title='Order Book Depth',
            template='plotly_dark',
            barmode='overlay'
        )

        return fig

# Example usage
if __name__ == "__main__":
    visualizer = EnhancedVisualizer()

    # Sample data
    pnl_data = pd.DataFrame({
        'pnl': np.random.normal(0.001, 0.02, 100)
    }, index=pd.date_range('2024-01-01', periods=100, freq='1min'))

    fig = visualizer.create_pnl_chart(pnl_data)
    fig.show()
