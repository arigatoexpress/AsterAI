"use client";
import { gql, useQuery } from '@apollo/client';
import client from '@/lib/apolloClient';
import { ApolloProvider } from '@apollo/client';

const MARKETS = gql`
	query Markets { markets { symbol price } }
`;

function MarketsInner() {
	const { data, loading, error } = useQuery(MARKETS);
	if (loading) return <div style={{padding:24}}>Loading…</div>;
	if (error) return <div style={{padding:24,color:'#f66'}}>Error: {error.message}</div>;
	return (
		<div style={{padding:24}}>
			<h2 style={{marginBottom:16}}>Markets</h2>
			<ul>
				{(data?.markets ?? []).map((m: any) => (
					<li key={m.symbol}>{m.symbol}: {m.price}</li>
				))}
			</ul>
		</div>
	);
}

export default function MarketsPage() {
	return (
		<ApolloProvider client={client}>
			<MarketsInner />
		</ApolloProvider>
	);
}


