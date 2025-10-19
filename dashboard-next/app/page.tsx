import Link from 'next/link';
import InfinitySymbol from '@/components/InfinitySymbol';

export default function HomePage() {
	return (
		<main style={{display:'flex',alignItems:'center',justifyContent:'center',minHeight:'100vh',background:'#0b0c10'}}>
			<div style={{textAlign:'center'}}>
				<h1 style={{color:'#c5c6c7',marginBottom:24,letterSpacing:1}}>AsterAI</h1>
				<InfinitySymbol size={140} />
				<div style={{marginTop:32, display:'flex', gap:16, justifyContent:'center'}}>
					<Link href="/markets" style={{padding:'10px 16px',border:'1px solid #444',borderRadius:8,color:'#c5c6c7'}}>Markets</Link>
					<Link href="/positions" style={{padding:'10px 16px',border:'1px solid #444',borderRadius:8,color:'#c5c6c7'}}>Positions</Link>
				</div>
			</div>
		</main>
	);
}


