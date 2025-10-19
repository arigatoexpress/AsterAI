export default function HelpPage() {
	return (
		<div style={{padding:24, maxWidth:900}}>
			<h2>Help</h2>
			<ol>
				<li>Overview shows high-level status and positions.</li>
				<li>Markets lists key symbols and prices.</li>
				<li>Positions shows open trades; Journal lists all executed trades with PnL.</li>
				<li>Insights provides actionable signals and confidence.</li>
				<li>GPU integrates performance charts; Logs links to Cloud Run logs.</li>
			</ol>
			<p>Use Presets to quickly switch risk profiles. All data auto-refreshes every few seconds.</p>
		</div>
	);
}


