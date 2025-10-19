"use client";
import { gql, useQuery } from '@apollo/client';
import client from '@/lib/apolloClient';
import { ApolloProvider } from '@apollo/client';

const POSITIONS = gql`
	query Positions { positions { id symbol qty entryPrice } }
`;

function PositionsInner() {
	const { data, loading, error } = useQuery(POSITIONS);
	if (loading) return <div style={{padding:24}}>Loading…</div>;
	if (error) return <div style={{padding:24,color:'#f66'}}>Error: {error.message}</div>;
	const items = data?.positions ?? [];
	return (
		<div style={{padding:24}}>
			<h2 style={{marginBottom:16}}>Positions</h2>
			{items.length === 0 ? (
				<div>No open positions.</div>
			) : (
				<table>
					<thead><tr><th>Symbol</th><th>Qty</th><th>Entry</th></tr></thead>
					<tbody>
						{items.map((p: any) => (
							<tr key={p.id}><td>{p.symbol}</td><td>{p.qty}</td><td>{p.entryPrice}</td></tr>
						))}
					</tbody>
				</table>
			)}
		</div>
	);
}

export default function PositionsPage() {
	return (
		<ApolloProvider client={client}>
			<PositionsInner />
		</ApolloProvider>
	);
}


