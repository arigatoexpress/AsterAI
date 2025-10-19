import type { Metadata } from 'next';
import Link from 'next/link';
import './globals.css';

export const metadata: Metadata = {
	title: 'AsterAI Dashboard',
	description: 'Next-gen trading observability and control',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
	return (
		<html lang="en">
			<body>
				<header style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'12px 20px',borderBottom:'1px solid #222',background:'#0b0c10'}}>
					<div style={{display:'flex',alignItems:'center',gap:12}}>
						<strong style={{color:'#c5c6c7'}}>AsterAI</strong>
					</div>
					<nav style={{display:'flex',gap:12}}>
						<Link href="/overview" style={{color:'#c5c6c7'}}>Overview</Link>
						<Link href="/markets" style={{color:'#c5c6c7'}}>Markets</Link>
						<Link href="/positions" style={{color:'#c5c6c7'}}>Positions</Link>
						<Link href="/journal" style={{color:'#c5c6c7'}}>Journal</Link>
						<Link href="/insights" style={{color:'#c5c6c7'}}>Insights</Link>
						<Link href="/gpu" style={{color:'#c5c6c7'}}>GPU</Link>
						<Link href="/logs" style={{color:'#c5c6c7'}}>Logs</Link>
						<Link href="/settings" style={{color:'#c5c6c7'}}>Settings</Link>
						<Link href="/help" style={{color:'#c5c6c7'}}>Help</Link>
						<Link href="/presets" style={{color:'#c5c6c7'}}>Presets</Link>
					</nav>
				</header>
				{children}
			</body>
		</html>
	);
}


