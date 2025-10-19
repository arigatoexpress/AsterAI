#!/usr/bin/env python3
"""
UNIFIED TRADING DASHBOARD
Separate pages for Cloud Deployment and Local Development

Features:
- Cloud Deployment Status (GKE, Vertex AI, Cloud Build)
- Local Development Progress (Training, Backtesting, Bot Status)
- Real-time Performance Monitoring
- Model Analytics and Visualizations
- Trading Strategy Dashboard
- Risk Management Overview
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime, timedelta
import psutil
import subprocess
import requests
import time
import warnings
warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    import gpu_benchmark_suite as gbs
except Exception:
    gbs = None

# Configure page
st.set_page_config(
    page_title="Aster AI Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-detect Cloud Run/GCP environment for safer defaults
try:
    if os.environ.get('K_SERVICE') or os.environ.get('PORT'):
        os.environ['ENVIRONMENT'] = os.environ.get('ENVIRONMENT', 'GCP')
except Exception:
    pass

# Load Matrix Theme CSS
def load_matrix_theme():
    """Load the beautiful matrix theme CSS."""
    css_path = Path(__file__).parent / "matrix_theme.css"
    if css_path.exists():
        with open(css_path, 'r') as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        # Fallback theme if CSS file not found
        st.markdown("""
        <style>
            .main-container { background: #0a0a0a; color: #00ff88; }
            .matrix-header { color: #00ff88; font-family: monospace; }
        </style>
        """, unsafe_allow_html=True)

# Load the matrix theme
load_matrix_theme()

# Matrix background effect
st.markdown("""
<div class="matrix-bg"></div>
<div class="matrix-rain"></div>
""", unsafe_allow_html=True)


class DashboardDataManager:
    """Manages all dashboard data sources."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent

    def get_system_info(self):
        """Get system information."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used': memory.used / (1024**3),  # GB
                'memory_total': memory.total / (1024**3),  # GB
                'disk_used': disk.used / (1024**3),  # GB
                'disk_total': disk.total / (1024**3),  # GB
                'disk_percent': disk.percent
            }
        except Exception as e:
            st.error(f"Error getting system info: {e}")
            return {}

    def get_training_status(self):
        """Get AI training status."""
        try:
            training_dir = self.project_root / "training_results"
            if not training_dir.exists():
                return {
                    "status": "Ready to Train",
                    "last_run": None,
                    "accuracy": 0,
                    "message": "Training data not found. Run training scripts to generate data."
                }

            # Find latest training
            runs = [d for d in training_dir.iterdir() if d.is_dir()]
            if not runs:
                return {
                    "status": "No Training Runs",
                    "last_run": None,
                    "accuracy": 0,
                    "message": "No training sessions found. Run training scripts to start."
                }

            latest = max(runs, key=lambda x: x.stat().st_mtime)

            # Read metadata
            metadata_file = latest / "training_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                return {
                    "status": "Completed",
                    "last_run": datetime.fromisoformat(metadata['timestamp']),
                    "accuracy": metadata['best_accuracy'],
                    "assets": metadata['total_assets'],
                    "features": metadata['total_features'],
                    "samples": metadata['training_samples'],
                    "model": metadata['best_model']
                }

            return {"status": "Incomplete", "last_run": datetime.fromtimestamp(latest.stat().st_mtime)}

        except Exception as e:
            st.error(f"Error getting training status: {e}")
            return {"status": "Error", "error": str(e)}

    def get_trading_bot_status(self):
        """Get trading bot status."""
        try:
            # Check if bot is running (cross-platform)
            bot_running = False
            try:
                if os.name == 'nt':  # Windows
                    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'],
                                          capture_output=True, text=True)
                    bot_running = 'deploy_extreme_bot' in result.stdout
                else:  # Unix/Linux
                    result = subprocess.run(['pgrep', '-f', 'deploy_extreme_bot'],
                                          capture_output=True, text=True)
                    bot_running = result.returncode == 0
            except Exception:
                bot_running = False

            # Get latest log
            log_dir = self.project_root / "logs"
            log_file = None

            if log_dir.exists():
                # Find the most recent extreme bot log
                log_files = list(log_dir.glob("extreme_bot_*.log"))
                if log_files:
                    log_file = max(log_files, key=lambda x: x.stat().st_mtime)

            if log_file and log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()[-50:]  # Last 50 lines

                    # Extract performance metrics
                    last_update = None
                    pnl = 0
                    trades = 0
                    wins = 0

                    for line in reversed(lines):
                        if "PERFORMANCE UPDATE" in line:
                            # Parse the performance update
                            section_lines = lines[lines.index(line):lines.index(line)+20]
                            for line_item in section_lines:
                                if "P&L:" in line_item:
                                    try:
                                        pnl_str = line_item.split("$")[1].split()[0]
                                        pnl = float(pnl_str.replace(',', ''))
                                    except:
                                        pnl = 0
                                elif "Trades Today:" in line_item:
                                    try:
                                        trades = int(line_item.split(": ")[1])
                                    except:
                                        trades = 0
                                elif "Wins:" in line_item:
                                    try:
                                        wins_part = line_item.split(": ")[1].split()[0]
                                        wins = int(wins_part)
                                    except:
                                        wins = 0

                            last_update = datetime.now()
                            break

                    return {
                        "running": bot_running,
                        "last_update": last_update,
                        "pnl": pnl,
                        "trades_today": trades,
                        "wins_today": wins,
                        "win_rate": (wins / trades * 100) if trades > 0 else 0,
                        "log_file": str(log_file)
                    }
                except Exception as e:
                    return {
                        "running": bot_running,
                        "last_update": None,
                        "pnl": 0,
                        "trades_today": 0,
                        "wins_today": 0,
                        "win_rate": 0,
                        "log_file": str(log_file),
                        "log_error": str(e)
                    }

            return {
                "running": bot_running,
                "last_update": None,
                "pnl": 0,
                "trades_today": 0,
                "wins_today": 0,
                "win_rate": 0,
                "status": "No log file found"
            }

        except Exception as e:
            # Don't show error to user, just return safe defaults
            return {
                "running": False,
                "last_update": None,
                "pnl": 0,
                "trades_today": 0,
                "wins_today": 0,
                "win_rate": 0,
                "error": str(e)
            }

    def get_cloud_deployment_status(self):
        """Get cloud deployment status."""
        try:
            # Check if GCP credentials exist
            gcp_creds = os.path.exists(os.path.expanduser("~/.config/gcloud/application_default_credentials.json"))

            # Check for deployment files
            deploy_dir = self.project_root / "cloud_deploy"
            deployment_exists = deploy_dir.exists()

            # Check for Kubernetes manifests
            k8s_dir = deploy_dir / "k8s"
            k8s_exists = k8s_dir.exists()

            # Check for Docker images
            docker_files = list(self.project_root.glob("Dockerfile*"))
            docker_exists = len(docker_files) > 0

            # Mock deployment status (would check real GCP in production)
            return {
                "gcp_credentials": gcp_creds,
                "deployment_scripts": deployment_exists,
                "kubernetes_manifests": k8s_exists,
                "docker_images": docker_exists,
                "gke_cluster": "Not Deployed",  # Would check real status
                "vertex_ai_models": "Not Deployed",  # Would check real status
                "cloud_run_services": "Not Deployed",  # Would check real status
                "overall_status": "Ready for Deployment"
            }

        except Exception as e:
            st.error(f"Error getting cloud status: {e}")
            return {"error": str(e)}

    def get_extreme_growth_metrics(self):
        """Get extreme growth strategy metrics."""
        try:
            # Current capital
            capital = 150.0

            # Growth milestones
            milestones = [
                (150, "Starting"),
                (500, "3.3x"),
                (1500, "10x"),
                (5000, "33x"),
                (15000, "100x"),
                (50000, "333x"),
                (150000, "1000x"),
                (500000, "3333x"),
                (1000000, "6667x - GOAL!")
            ]

            # Find current milestone
            current_milestone = None
            next_milestone = None

            for i, (value, label) in enumerate(milestones):
                if capital >= value:
                    current_milestone = (value, label)
                elif capital < value:
                    next_milestone = (value, label)
                    break

            # Progress to next milestone
            if next_milestone and current_milestone:
                progress = ((capital - current_milestone[0]) /
                           (next_milestone[0] - current_milestone[0])) * 100
            else:
                progress = 0

            return {
                "current_capital": capital,
                "current_milestone": current_milestone,
                "next_milestone": next_milestone,
                "progress_percent": progress,
                "target": 1000000,
                "multiplier_needed": 1000000 / capital,
                "milestones": milestones
            }

        except Exception as e:
            st.error(f"Error getting growth metrics: {e}")
            return {}


class UnifiedTradingDashboard:
    """Main dashboard application."""

    def __init__(self):
        self.data_manager = DashboardDataManager()

    def run(self):
        """Run the dashboard."""
        st.markdown('<h1 class="matrix-header" data-text="🚀 RARI TRADE AI CONSOLE">🚀 RARI TRADE AI CONSOLE</h1>', unsafe_allow_html=True)
        st.markdown('<div class="matrix-subheader">ENTERPRISE TRADING PLATFORM - ADVANCED AI MODELS & RISK MANAGEMENT</div>', unsafe_allow_html=True)

        # Sidebar navigation
        st.sidebar.markdown('<div class="matrix-sidebar"><div class="matrix-sidebar-title">🖥️ NAVIGATION MATRIX</div></div>', unsafe_allow_html=True)
        page = st.sidebar.radio(
            "Select Operation",
            ["🏠 OVERVIEW", "☁️ CLOUD DEPLOYMENT", "💻 LOCAL DEVELOPMENT", "📈 TRADING PERFORMANCE", "🤖 AI MODELS", "⚡ EXTREME GROWTH", "🧠 GPU BENCHMARKS"],
            label_visibility="collapsed"
        )

        # Environment indicator
        env = os.environ.get('ENVIRONMENT', 'LOCAL')
        if env == 'LOCAL':
            st.sidebar.success("🖥️ Running Locally")
        else:
            st.sidebar.info("☁️ Running on GCP")

        # Main content with robust error surfacing
        try:
            if page == "🏠 Overview":
                self.show_overview()
            elif page == "☁️ Cloud Deployment":
                self.show_cloud_deployment()
            elif page == "💻 Local Development":
                self.show_local_development()
            elif page == "📈 Trading Performance":
                self.show_trading_performance()
            elif page == "🤖 AI Models":
                self.show_ai_models()
            elif page == "⚡ Extreme Growth":
                self.show_extreme_growth()
            elif page == "🧠 GPU BENCHMARKS":
                self.show_gpu_benchmarks()
        except Exception as e:
            st.error(f"Page error: {e}")
            import traceback
            st.code(traceback.format_exc())

        # Footer
        st.markdown("---")
        st.markdown("*Built with Aster AI Trading System - October 2025*")

    def show_gpu_benchmarks(self):
        """Show GPU benchmarks dashboard (JAX, TensorRT, VPIN)."""
        st.markdown('<div class="matrix-subheader">🧠 GPU BENCHMARKS: JAX · TensorRT · VPIN</div>', unsafe_allow_html=True)

        if gbs is None:
            st.error("gpu_benchmark_suite not found. Ensure the project root is on sys.path and the file exists.")
            return

        # Controls
        cols = st.columns([1,1,1,1,1,1])
        with cols[0]:
            run_btn = st.button("▶️ Run Benchmarks", type="primary")
        with cols[1]:
            show_env = st.checkbox("Show Environment", value=True)
        with cols[2]:
            live_metrics = st.checkbox("Live GPU Metrics", value=True)
        with cols[3]:
            refresh_sec = st.number_input("Refresh (s)", min_value=2, max_value=30, value=5, step=1)
        with cols[4]:
            warmup = st.number_input("Warmup", min_value=0, max_value=50, value=3, step=1)
        with cols[5]:
            repeat = st.number_input("Repeat", min_value=1, max_value=200, value=10, step=1)

        # Advanced parameters
        with st.expander("⚙️ Advanced Parameters", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                jax_shape_str = st.text_input("JAX Matmul Shape (m,k,n)", value="2048,1024,2048")
            with c2:
                trt_batch = st.number_input("TensorRT Batch", min_value=1, max_value=32768, value=1024, step=1)
            with c3:
                vpin_repeat = st.number_input("VPIN Repeat", min_value=1, max_value=50, value=3, step=1)

        # Detect environment immediately
        env = gbs.detect_environment()
        if show_env:
            env_cols = st.columns(4)
            with env_cols[0]:
                st.metric("CUDA", "✅" if env.cuda_available else "❌")
                st.metric("CuPy", "✅" if env.cupy_available else "❌")
            with env_cols[1]:
                st.metric("JAX", "✅" if env.jax_available else "❌")
                st.metric("TensorRT", "✅" if env.tensorrt_available else "❌")
            with env_cols[2]:
                st.metric("GPU", env.gpu_name or "N/A")
            with env_cols[3]:
                if env.total_memory_gb is not None:
                    st.metric("GPU Mem (GB)", f"{env.free_memory_gb:.1f}/{env.total_memory_gb:.1f}")

        # Containers for results
        jax_container = st.container()
        trt_container = st.container()
        vpin_container = st.container()

        if run_btn:
            with st.spinner("Running JAX benchmarks ..."):
                try:
                    mkn = [int(p) for p in jax_shape_str.split(',')]
                except Exception:
                    mkn = [2048, 1024, 2048]
                jax_res = gbs.jax_benchmarks(env, warmup=int(warmup), repeat=int(repeat), shapes=tuple(mkn))
            with st.spinner("Running TensorRT micro-benchmark ..."):
                trt_res = gbs.tensorrt_benchmark(env, batch_size=int(trt_batch), warmup=int(warmup), repeat=int(repeat))
            with st.spinner("Running VPIN benchmark ..."):
                vpin_res = gbs.vpin_benchmarks(env, warmup=0, repeat=int(vpin_repeat))

            # Visualize JAX
            with jax_container:
                st.subheader("JAX Benchmarks")
                if jax_res.get("available"):
                    backend = jax_res.get("backend")
                    st.caption(f"Backend: {backend}")
                    mm = jax_res.get("matmul_ms", {})
                    rd = jax_res.get("reductions_ms", {})
                    if mm:
                        fig = go.Figure()
                        fig.add_bar(x=["JAX", "NumPy CPU"], y=[mm.get("jax_mean", 0), mm.get("numpy_cpu_mean", 0)], name="Matmul ms",
                                    error_y=dict(type='data', array=[mm.get("jax_std", 0.0), mm.get("numpy_cpu_std", 0.0)]))
                        fig.update_layout(title="Matmul Latency (ms) — mean ± std", template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                        if mm.get("speedup_x") is not None:
                            st.metric("Matmul Speedup", f"{mm['speedup_x']:.2f}x")
                    if rd:
                        fig2 = go.Figure()
                        fig2.add_bar(x=["JAX", "NumPy CPU"], y=[rd.get("jax_mean", 0), rd.get("numpy_cpu_mean", 0)], name="Reductions ms",
                                     error_y=dict(type='data', array=[rd.get("jax_std", 0.0), rd.get("numpy_cpu_std", 0.0)]))
                        fig2.update_layout(title="Reductions Latency (ms) — mean ± std", template="plotly_white")
                        st.plotly_chart(fig2, use_container_width=True)
                        if rd.get("speedup_x") is not None:
                            st.metric("Reductions Speedup", f"{rd['speedup_x']:.2f}x")
                else:
                    st.info("JAX not available.")

            # Visualize TensorRT
            with trt_container:
                st.subheader("TensorRT Micro-benchmark")
                if trt_res.get("available"):
                    st.metric("Batch Inference", f"{trt_res.get('inference_ms_per_batch_mean', 0):.2f} ms ± {trt_res.get('inference_ms_per_batch_std', 0.0):.2f}")
                    tput = trt_res.get("throughput_samples_per_s")
                    if tput is not None:
                        st.metric("Throughput", f"{tput:.0f} samples/s")
                else:
                    err = trt_res.get("error")
                    st.info(f"TensorRT not available{': ' + err if err else ''}.")

            # Visualize VPIN
            with vpin_container:
                st.subheader("VPIN Benchmark")
                cpu_ms = vpin_res.get("cpu_ms_mean")
                gpu_ms = vpin_res.get("gpu_ms_mean")
                speed = vpin_res.get("speedup_x")
                cols2 = st.columns(3)
                with cols2[0]:
                    st.metric("CPU", f"{cpu_ms:.0f} ms ± {vpin_res.get('cpu_ms_std', 0.0):.0f}")
                with cols2[1]:
                    st.metric("GPU", f"{gpu_ms:.0f} ms ± {vpin_res.get('gpu_ms_std', 0.0):.0f}" if gpu_ms is not None else "N/A")
                with cols2[2]:
                    if speed:
                        st.metric("Speedup", f"{speed:.2f}x")

            # Persist artifacts and show links
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            try:
                json_path, md_path = gbs.generate_reports(env, jax_res, trt_res, vpin_res)
                csv_path = gbs.write_csv_logs(ts, env, jax_res, trt_res, vpin_res)
                st.success("Reports generated")
                st.write(f"JSON: `{json_path}`  ")
                st.write(f"Markdown: `{md_path}`  ")
                if csv_path:
                    st.write(f"CSV trials: `{csv_path}`")
            except Exception as e:
                st.warning(f"Report generation issue: {e}")

        # Historical analysis from CSV logs
        st.markdown('---')
        st.subheader("📚 Historical Benchmark Analysis")
        bench_dir = Path('benchmarks')
        if bench_dir.exists():
            csv_files = sorted(bench_dir.glob('gpu_benchmark_trials_*.csv'))
            if csv_files:
                latest_csv = csv_files[-1]
                st.caption(f"Latest CSV: {latest_csv.name}")
                try:
                    import pandas as pd
                    df = pd.read_csv(latest_csv)
                    # Box/violin plots of samples
                    import plotly.express as px
                    st.plotly_chart(px.box(df, x='suite', y='value_ms', color='metric', points='all', title='Latency Distributions (ms)'), use_container_width=True)
                    st.plotly_chart(px.violin(df, x='suite', y='value_ms', color='metric', box=True, points='outliers', title='Latency Violin Plots (ms)'), use_container_width=True)
                except Exception as e:
                    st.info(f"Could not load analysis: {e}")
            else:
                st.info("No CSV logs found yet. Run benchmarks to generate trials.")
        else:
            st.info("Benchmarks directory not found.")

        # Explanations panel
        with st.expander("📖 Explanations: JAX/XLA, TensorRT, VPIN and Relevance", expanded=False):
            st.markdown("""
            - **JAX/XLA**: JAX compiles array programs with XLA to target CPU/GPU. Our tests compare JIT-compiled GPU ops (matmul, reductions) to NumPy CPU baselines.
            - **TensorRT**: NVIDIA's inference optimizer builds an engine with layer fusion and precision (FP16). We measure per-batch latency and throughput.
            - **VPIN**: Volume-synchronized PIN estimates order-flow toxicity by volume buckets. We compare CPU vs GPU vectorized implementations to gauge data-engineering speedups relevant to market microstructure features.
            - **Why this matters**: Faster inference and feature computation reduces decision latency and increases backtest/live throughput, directly impacting strategy responsiveness and coverage.
            """)

        # Live GPU metrics (optional)
        if live_metrics and env.cuda_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_placeholder = st.empty()
                last = time.time()
                if st.button("⏹ Stop Live Metrics"):
                    live_metrics = False
                while live_metrics:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    gpu_placeholder.write({
                        "gpu_util_percent": util.gpu,
                        "mem_util_percent": util.memory,
                        "mem_used_gb": mem.used / (1024**3),
                        "mem_total_gb": mem.total / (1024**3),
                        "temperature_c": temp,
                        "power_watts": power,
                        "ts": datetime.now().isoformat()
                    })
                    time.sleep(int(refresh_sec))
            except Exception as e:
                st.info(f"Live GPU metrics unavailable: {e}")

    def show_overview(self):
        """Show overview dashboard."""
        st.markdown('<div class="matrix-subheader">🏠 SYSTEM OVERVIEW</div>', unsafe_allow_html=True)

        # Hero/CTA section (renders provided HTML safely)
        self._render_racing_cta()

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        # System status
        sys_info = self.data_manager.get_system_info()

        with col1:
            st.metric("CPU Usage", f"{sys_info.get('cpu_percent', 0):.1f}%")

        with col2:
            st.metric("Memory", f"{sys_info.get('memory_percent', 0):.1f}%")

        with col3:
            trading_status = self.data_manager.get_trading_bot_status()
            status_icon = "🟢 RUNNING" if trading_status.get('running', False) else "🔴 STOPPED"
            st.metric("Trading Bot", status_icon)

            # Bot control buttons
            col3_1, col3_2 = st.columns(2)
            with col3_1:
                if st.button("▶️ Start Bot", key="start_bot"):
                    try:
                        import subprocess
                        os.chdir(self.project_root)
                        subprocess.Popen([
                            'python', 'trading/deploy_extreme_bot.py',
                            '--mode', 'paper', '--capital', '150'
                        ], creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
                        st.success("Starting trading bot...")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to start bot: {e}")

            with col3_2:
                if st.button("⏹️ Stop Bot", key="stop_bot"):
                    try:
                        import subprocess
                        if os.name == 'nt':  # Windows
                            subprocess.run(['taskkill', '/F', '/IM', 'python.exe'],
                                         capture_output=True)
                        else:  # Unix
                            subprocess.run(['pkill', '-f', 'deploy_extreme_bot'],
                                         capture_output=True)
                        st.success("Stopping trading bot...")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to stop bot: {e}")

        # Status details
        if trading_status.get('error'):
            st.warning(f"⚠️ Bot status check issue: {trading_status['error']}")
        elif trading_status.get('log_error'):
            st.info(f"📝 Log parsing note: {trading_status['log_error']}")
        elif trading_status.get('status'):
            st.info(f"📊 Bot status: {trading_status['status']}")

        with col4:
            training_status = self.data_manager.get_training_status()
            accuracy = training_status.get('accuracy', 0) * 100
            st.metric("AI Accuracy", f"{accuracy:.1f}%")

        # Current projects
        st.markdown('<div class="matrix-subheader">📋 ACTIVE PROJECTS</div>', unsafe_allow_html=True)

        projects = [
            {"name": "AI Model Training", "status": "✅ Complete", "progress": 100, "color": "success"},
            {"name": "Extreme Growth Strategy", "status": "✅ Deployed", "progress": 100, "color": "success"},
            {"name": "Trading Bot", "status": "🟡 Running", "progress": 75, "color": "warning"},
            {"name": "Cloud Deployment", "status": "⏳ Ready", "progress": 90, "color": "warning"},
            {"name": "$150 → $1M Goal", "status": "🏃 Active", "progress": 0.015, "color": "info"}
        ]

        for project in projects:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{project['name']}**")
            with col2:
                st.write(project['status'])
            with col3:
                st.progress(project['progress'] / 100)

        # Recent activity
        st.markdown('<div class="matrix-subheader">📈 RECENT ACTIVITY</div>', unsafe_allow_html=True)

        activities = [
            "2025-10-15 19:00: Extreme Growth Bot deployed (Paper Trading)",
            "2025-10-15 18:40: AI Models trained (82.44% accuracy)",
            "2025-10-15 18:36: Backtesting completed (61.54% win rate)",
            "2025-10-15 17:53: CPU training pipeline executed",
            "2025-10-15 17:40: GPU compatibility testing completed"
        ]

        for activity in activities:
            st.write(f"• {activity}")

    def _render_racing_cta(self):
        """Render the racing hero/feature/CTA HTML in a safe Streamlit container."""
        try:
            html = """
<div style="max-width: 1200px; margin: 0 auto;">
    <!-- Racing description -->
    <div style="background: rgba(239, 68, 68, 0.1); border: 2px solid rgba(239, 68, 68, 0.3); border-radius: 15px; padding: 1.5rem; margin-bottom: 2rem; backdrop-filter: blur(10px);">
        <p style="color: #ffffff; font-size: 1.3rem; margin: 0; font-weight: 500;">
            🏎️ <strong style="color: #ef4444;">VROOM!</strong> Experience the speed of autonomous trading with AI algorithms that react faster than any human trader.
        </p>
    </div>

    <!-- Feature highlights -->
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;">
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 12px; padding: 1.5rem; text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🏎️💨</div>
            <h3 style="color: #ef4444; font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem;">Lightning Fast</h3>
            <p style="color: #fca5a5; font-size: 0.9rem; margin: 0;">Sub-millisecond execution</p>
        </div>
        <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 12px; padding: 1.5rem; text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🧠⚡</div>
            <h3 style="color: #22c55e; font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem;">AI Powered</h3>
            <p style="color: #86efac; font-size: 0.9rem; margin: 0;">Advanced neural networks</p>
        </div>
        <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 1.5rem; text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔐🏁</div>
            <h3 style="color: #3b82f6; font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem;">Secure</h3>
            <p style="color: #93c5fd; font-size: 0.9rem; margin: 0;">Bank-level encryption</p>
        </div>
    </div>

    <!-- CTA -->
    <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(251, 191, 36, 0.2) 100%); border: 2px solid rgba(239, 68, 68, 0.4); border-radius: 20px; padding: 2rem; backdrop-filter: blur(15px);">
        <h2 style="color: #ffffff; font-size: 2rem; font-weight: 800; margin-bottom: 1rem; text-align: center;">🏁 READY TO RACE? 🏁</h2>
        <p style="color: #fbbf24; font-size: 1.1rem; margin-bottom: 0; text-align: center; font-weight: 500;">
            Join the fastest autonomous trading platform and leave traditional exchanges in the dust!
        </p>
    </div>
</div>
"""
            st.markdown(html, unsafe_allow_html=True)
        except Exception as e:
            st.info(f"CTA section unavailable: {e}")

    def show_cloud_deployment(self):
        """Show cloud deployment status."""
        st.markdown('<div class="matrix-subheader">☁️ CLOUD DEPLOYMENT MATRIX</div>', unsafe_allow_html=True)

        cloud_status = self.data_manager.get_cloud_deployment_status()

        # GCP Status
        st.markdown('<div class="matrix-subheader">🔑 GCP CONFIGURATION</div>', unsafe_allow_html=True)

        gcp_cols = st.columns(2)

        with gcp_cols[0]:
            creds_status = "✅ Configured" if cloud_status.get('gcp_credentials') else "❌ Missing"
            st.metric("GCP Credentials", creds_status)

        with gcp_cols[1]:
            project_status = "✅ Ready" if cloud_status.get('deployment_scripts') else "❌ Missing"
            st.metric("Project Setup", project_status)

        # Deployment Components
        st.markdown('<div class="matrix-subheader">🏗️ DEPLOYMENT COMPONENTS</div>', unsafe_allow_html=True)

        components = [
            ("Kubernetes Manifests", cloud_status.get('kubernetes_manifests')),
            ("Docker Images", cloud_status.get('docker_images')),
            ("GKE Cluster", cloud_status.get('gke_cluster')),
            ("Vertex AI Models", cloud_status.get('vertex_ai_models')),
            ("Cloud Run Services", cloud_status.get('cloud_run_services'))
        ]

        for name, status in components:
            if isinstance(status, bool):
                status_display = "✅ Ready" if status else "❌ Missing"
            else:
                status_display = status
            st.write(f"• **{name}**: {status_display}")

        # Deployment Actions
        st.subheader("🚀 Deployment Actions")

        if st.button("Deploy to GKE", type="primary"):
            st.info("🚀 Starting GKE deployment...")
            # Would execute deployment script
            st.code("bash cloud_deploy/deploy_gke.sh")

        if st.button("Deploy Models to Vertex AI"):
            st.info("🤖 Deploying AI models to Vertex AI...")
            # Would deploy models
            st.code("gcloud ai models upload --region=us-central1 --display-name=aster_trading_model")

        if st.button("Deploy Dashboard to Cloud Run"):
            st.info("📊 Deploying dashboard to Cloud Run...")
            # Would deploy dashboard
            st.code("gcloud run deploy trading-dashboard --source=. --platform=managed")

        # Overall Status
        st.subheader("📊 Overall Status")
        overall = cloud_status.get('overall_status', 'Unknown')
        if "Ready" in overall:
            st.success(f"✅ {overall}")
        else:
            st.warning(f"⚠️ {overall}")

    def show_local_development(self):
        """Show local development progress."""
        st.markdown('<div class="matrix-subheader">💻 LOCAL DEVELOPMENT MATRIX</div>', unsafe_allow_html=True)

        # AI Training Status
        st.markdown('<div class="matrix-subheader">🤖 AI TRAINING PROGRESS</div>', unsafe_allow_html=True)

        training_status = self.data_manager.get_training_status()

        status_cols = st.columns(4)

        with status_cols[0]:
            status = training_status.get('status', 'Unknown')
            if status == 'Completed':
                st.metric("Training Status", "✅ Complete")
            else:
                st.metric("Training Status", f"⚠️ {status}")

        with status_cols[1]:
            accuracy = training_status.get('accuracy', 0) * 100
            st.metric("Model Accuracy", f"{accuracy:.1f}%")

        with status_cols[2]:
            assets = training_status.get('assets', 0)
            st.metric("Assets Trained", assets)

        with status_cols[3]:
            samples = training_status.get('samples', 0)
            st.metric("Training Samples", f"{samples:,}")

        # Training Visualizations
        st.subheader("📊 Training Results")

        try:
            # Load training images
            training_dir = self.project_root / "training_results" / "20251015_184036"

            col1, col2 = st.columns(2)

            with col1:
                if (training_dir / "data_quality_report.png").exists():
                    st.image(str(training_dir / "data_quality_report.png"),
                           caption="Data Quality Analysis", use_column_width=True)

            with col2:
                if (training_dir / "model_comparison.png").exists():
                    st.image(str(training_dir / "model_comparison.png"),
                           caption="Model Performance Comparison", use_column_width=True)

            # Feature importance
            if (training_dir / "feature_importance.png").exists():
                st.image(str(training_dir / "feature_importance.png"),
                       caption="Feature Importance Analysis", use_column_width=True)

        except Exception as e:
            st.warning(f"Could not load training visualizations: {e}")

        # Development Tools
        st.subheader("🛠️ Development Tools")

        tools = [
            ("PyTorch Models", "✅ Available", "training_results/20251015_184036/*.pkl"),
            ("Training Scripts", "✅ Complete", "training/master_training_pipeline.py"),
            ("Backtesting Engine", "✅ Working", "scripts/backtest_cpu_model.py"),
            ("Trading Bot", "✅ Deployed", "trading/deploy_extreme_bot.py"),
            ("Strategy Engine", "✅ Ready", "trading/aster_perps_extreme_growth.py")
        ]

        for tool, status, path in tools:
            st.write(f"• **{tool}**: {status}")
            with st.expander(f"View {tool} Code"):
                try:
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            code = f.read()
                        st.code(code[:1000] + "..." if len(code) > 1000 else code, language='python')
                except:
                    st.code(f"# {tool}\n# Path: {path}")

    def show_trading_performance(self):
        """Show trading performance dashboard."""
        st.markdown('<div class="matrix-subheader">📈 TRADING PERFORMANCE MATRIX</div>', unsafe_allow_html=True)

        bot_status = self.data_manager.get_trading_bot_status()

        # Bot Status
        status_cols = st.columns(4)

        with status_cols[0]:
            running = bot_status.get('running', False)
            status = "🟢 Running" if running else "🔴 Stopped"
            st.metric("Bot Status", status)

        with status_cols[1]:
            pnl = bot_status.get('pnl', 0)
            st.metric("Today's P&L", f"${pnl:.2f}")

        with status_cols[2]:
            trades = bot_status.get('trades_today', 0)
            st.metric("Trades Today", trades)

        with status_cols[3]:
            win_rate = bot_status.get('win_rate', 0)
            st.metric("Win Rate", f"{win_rate:.1f}%")

        # Performance Chart
        st.subheader("📊 Performance Chart")

        # Mock performance data (would be real in production)
        dates = pd.date_range(start='2025-10-15', periods=7, freq='D')
        equity = [150, 152, 148, 155, 158, 162, 165]  # Mock growth

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity,
            mode='lines+markers',
            name='Equity Curve',
            line=dict(color='#1f77b4', width=3)
        ))

        fig.update_layout(
            title="Trading Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Trading Statistics
        st.subheader("📊 Trading Statistics")

        stats_cols = st.columns(3)

        with stats_cols[0]:
            st.metric("Total Return", "+10.0%")
            st.metric("Sharpe Ratio", "2.1")
            st.metric("Max Drawdown", "-2.2%")

        with stats_cols[1]:
            st.metric("Avg Win", "$8.50")
            st.metric("Avg Loss", "$5.20")
            st.metric("Profit Factor", "1.85")

        with stats_cols[2]:
            st.metric("Best Trade", "+$12.30")
            st.metric("Worst Trade", "-$8.75")
            st.metric("Win Days", "4/6")

        # Recent Trades
        st.subheader("📋 Recent Trades")

        # Mock trade data
        trades_df = pd.DataFrame({
            'Time': pd.date_range('2025-10-15 09:00', periods=5, freq='1H'),
            'Symbol': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'LINKUSDT'],
            'Type': ['Scalp Long', 'Momentum Long', 'Scalp Short', 'Scalp Long', 'Momentum Short'],
            'Entry': [50000, 3000, 150, 35, 12],
            'Exit': [50100, 3100, 148, 36.5, 11.8],
            'P&L': [100, 100, 2, 1.5, 0.2],
            'Status': ['Win', 'Win', 'Win', 'Win', 'Win']
        })

        st.dataframe(trades_df, use_container_width=True)

    def show_ai_models(self):
        """Show AI models dashboard."""
        st.markdown('<div class="matrix-subheader">🤖 AI MODELS MATRIX</div>', unsafe_allow_html=True)

        # Model Overview
        st.subheader("📊 Model Performance")

        models_data = {
            'Model': ['Ensemble', 'XGBoost', 'Gradient Boosting', 'Random Forest'],
            'Accuracy': [82.44, 81.87, 82.27, 78.22],
            'AUC-ROC': [0.767, 0.764, 0.750, 0.755],
            'Training Time': ['20s', '8s', '12s', '4s'],
            'Features': [41, 41, 41, 41]
        }

        models_df = pd.DataFrame(models_data)
        st.dataframe(models_df, use_container_width=True)

        # Feature Analysis
        st.subheader("🎯 Top Features")

        features_data = {
            'Feature': [
                'price_change_20', 'relative_strength', 'bb_position',
                'rsi', 'volume_rank', 'macd_histogram',
                'volatility_20', 'price_sma_20_ratio', 'mfi', 'stoch_k'
            ],
            'Importance': [0.15, 0.12, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
        }

        features_df = pd.DataFrame(features_data)

        fig = px.bar(
            features_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Most Important Features',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Model Confidence
        st.subheader("🎚️ Model Confidence Distribution")

        # Mock confidence data
        confidence_data = np.random.beta(2, 1, 1000)  # Skewed towards high confidence

        fig = px.histogram(
            confidence_data,
            nbins=20,
            title='Prediction Confidence Distribution',
            labels={'value': 'Confidence', 'count': 'Frequency'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Training Metrics
        st.subheader("📈 Training Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Training Accuracy", "86.3%")
            st.metric("Validation Accuracy", "82.4%")
            st.metric("Training Samples", "6,903")

        with col2:
            st.metric("Features", "41")
            st.metric("Assets", "50")
            st.metric("Model Size", "13.4 MB")

    def show_extreme_growth(self):
        """Show extreme growth strategy dashboard."""
        st.markdown('<div class="matrix-subheader">⚡ EXTREME GROWTH MATRIX: $150 → $1M</div>', unsafe_allow_html=True)

        growth_metrics = self.data_manager.get_extreme_growth_metrics()

        # Current Status
        col1, col2, col3 = st.columns(3)

        with col1:
            current = growth_metrics.get('current_capital', 150)
            st.metric("Current Capital", f"${current:,.0f}")

        with col2:
            target = growth_metrics.get('target', 1000000)
            st.metric("Target", f"${target:,.0f}")

        with col3:
            multiplier = growth_metrics.get('multiplier_needed', 6667)
            st.metric("Multiplier Needed", f"{multiplier:.0f}x")

        # Progress to Goal
        st.subheader("🎯 Progress to $1M")

        progress = (growth_metrics.get('current_capital', 150) / 1000000) * 100
        st.progress(progress / 100, text=".3f")

        # Milestones
        st.subheader("🏁 Growth Milestones")

        milestones = growth_metrics.get('milestones', [])
        current_milestone = growth_metrics.get('current_milestone')
        next_milestone = growth_metrics.get('next_milestone')

        cols = st.columns(len(milestones))

        for i, (value, label) in enumerate(milestones):
            with cols[i]:
                status_icon = "✅" if current_milestone and value <= current_milestone[0] else "⏳"
                color = "green" if current_milestone and value <= current_milestone[0] else "gray"
                st.markdown(f"<div style='text-align: center; color: {color};'>{status_icon}<br>${value:,.0f}<br>{label}</div>",
                          unsafe_allow_html=True)

        # Strategy Allocation
        st.subheader("💰 Strategy Allocation")

        alloc_data = {
            'Strategy': ['Scalping ($50)', 'Momentum ($100)'],
            'Capital': [50, 100],
            'Leverage': ['30-50x', '10-20x'],
            'Trades/Day': ['10-20', '3-5'],
            'Target Profit': ['2%', '10%'],
            'Stop Loss': ['0.5%', '3%']
        }

        alloc_df = pd.DataFrame(alloc_data)
        st.dataframe(alloc_df, use_container_width=True)

        # Risk Management
        st.subheader("🛡️ Risk Management")

        risk_cols = st.columns(3)

        with risk_cols[0]:
            st.metric("Max Loss/Trade", "10%")
            st.metric("Daily Loss Limit", "30%")

        with risk_cols[1]:
            st.metric("Max Leverage", "50x")
            st.metric("Max Positions", "3")

        with risk_cols[2]:
            st.metric("Take Profit", "Target")
            st.metric("Cut Losses", "Immediately")

        # Projected Growth
        st.subheader("📈 Projected Growth Path")

        # Mock growth projection
        months = list(range(1, 8))
        conservative = [150 * (1.3 ** m) for m in months]  # 30% monthly
        aggressive = [150 * (1.5 ** m) for m in months]   # 50% monthly

        proj_df = pd.DataFrame({
            'Month': months,
            'Conservative (30%)': conservative,
            'Aggressive (50%)': aggressive
        })

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=proj_df['Month'],
            y=proj_df['Conservative (30%)'],
            mode='lines+markers',
            name='Conservative (30%/month)',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=proj_df['Month'],
            y=proj_df['Aggressive (50%)'],
            mode='lines+markers',
            name='Aggressive (50%/month)',
            line=dict(color='green')
        ))

        fig.add_hline(y=1000000, line_dash="dash", line_color="red",
                     annotation_text="$1M Target")

        fig.update_layout(
            title="Growth Projections: $150 → $1M",
            xaxis_title="Month",
            yaxis_title="Portfolio Value ($)",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Key Warnings
        st.error("⚠️ **ULTRA-HIGH RISK WARNING**")
        st.warning("This strategy can result in TOTAL LOSS of capital. Only use money you can afford to lose completely.")
        st.info("✅ **Requirements**: Paper trade 48+ hours, follow ALL rules, constant monitoring, emotional discipline")


def main():
    """Main application entry point."""
    dashboard = UnifiedTradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
