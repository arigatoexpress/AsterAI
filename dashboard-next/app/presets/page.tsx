"use client";
import { useState } from 'react';

const PRESETS = [
	{ name: 'Conservative', positionSizePct: 0.01, maxPositions: 1 },
	{ name: 'Balanced', positionSizePct: 0.03, maxPositions: 2 },
	{ name: 'Aggressive', positionSizePct: 0.05, maxPositions: 3 },
];

export default function PresetsPage() {
	const [sel, setSel] = useState(PRESETS[1]);
	return (
		<div style={{padding:24}}>
			<h2>Presets</h2>
			<div style={{display:'flex', gap:12}}>
				{PRESETS.map(p => (
					<button key={p.name} onClick={() => setSel(p)} style={{padding:'8px 12px',border:'1px solid #222',borderRadius:8,color:'#c5c6c7',background:'transparent'}}>
						{p.name}
					</button>
				))}
			</div>
			<div style={{marginTop:16}}>
				<div>Position Size %: {(sel.positionSizePct*100).toFixed(1)}%</div>
				<div>Max Positions: {sel.maxPositions}</div>
			</div>
			<p style={{opacity:0.8, marginTop:12}}>Apply these settings in your bot configuration to scale across machines consistently.</p>
		</div>
	);
}


