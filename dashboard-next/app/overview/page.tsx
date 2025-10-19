"use client";
import { gql, useQuery } from '@apollo/client';
import { ApolloProvider } from '@apollo/client';
import client from '@/lib/apolloClient';

const DASH = gql`
	query Dashboard {
		markets { symbol price }
		positions { id symbol qty entryPrice }
	}
`;

function OverviewInner() {
	const { data, loading, error } = useQuery(DASH, { pollInterval: 5000 });
	if (loading) return <div style={{padding:24}}>Loading…</div>;
	if (error) return <div style={{padding:24,color:'#f66'}}>Error: {error.message}</div>;
	const markets = data?.markets ?? [];
	const positions = data?.positions ?? [];
	return (
		<div style={{padding:24, display:'grid', gridTemplateColumns:'1fr 1fr', gap:16}}>
			<section style={{border:'1px solid #222', borderRadius:8, padding:16}}>
				<h3>Markets (Top)</h3>
				<ul style={{margin:0, paddingLeft:16}}>
					{markets.slice(0,8).map((m:any) => (
						<li key={m.symbol}>{m.symbol}: {m.price}</li>
					))}
				</ul>
			</section>
			<section style={{border:'1px solid #222', borderRadius:8, padding:16}}>
				<h3>Open Positions</h3>
				{positions.length === 0 ? <div>No open positions.</div> : (
					<table>
						<thead><tr><th>Symbol</th><th>Qty</th><th>Entry</th></tr></thead>
						<tbody>
							{positions.map((p:any) => (
								<tr key={p.id}><td>{p.symbol}</td><td>{p.qty}</td><td>{p.entryPrice}</td></tr>
							))}
						</tbody>
					</table>
				)}
			</section>
		</div>
	);
}

export default function OverviewPage() {
	return (
		<ApolloProvider client={client}>
			<OverviewInner />
		</ApolloProvider>
	);
}


